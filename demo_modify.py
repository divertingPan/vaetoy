"""
PC端交互Demo (MCU模拟升级版) - 模拟玩具在单片机上的实际运算行为
使用Pygame显示16x16 LED矩阵效果
特性：
1. 采用与单片机 (ESP32) 完全一致的潜空间高斯随机采样 (N(mu, sigma^2))
2. 连续按下相同数字时，会从当前字形缓缓变幻为该数字的另一副笔迹
3. 过渡时间统一为 45 帧 (约 0.75 秒 @ 60FPS)，采用 Smoothstep 缓动曲线
4. 同时支持主键盘 0-9 与数字小键盘 (NumPad 0-9)
"""

import pygame
import torch
import torch.nn.functional as F
import numpy as np
import os
import re
import sys

# ---- 复用模型定义 ----
LATENT_DIM = 16
IMAGE_SIZE = 16


class Encoder(torch.nn.Module):
    """
    16x16 -> Conv层 -> 潜空间 (mu, log_var)
    结构设计: 16x16x1 -> 8x8x24 -> 4x4x48 -> 2x2x64 -> 16维潜空间 (与 train_vae.py 100% 同步)
    """
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(1, 24, kernel_size=3, stride=2, padding=1)   # -> 8x8x24
        self.conv2 = torch.nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1)  # -> 4x4x48
        self.conv3 = torch.nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1)  # -> 2x2x64
        self.fc_mu = torch.nn.Linear(64 * 2 * 2, latent_dim)
        self.fc_logvar = torch.nn.Linear(64 * 2 * 2, latent_dim)

    def forward(self, x):
        h = F.relu(self.conv1(x))
        h = F.relu(self.conv2(h))
        h = F.relu(self.conv3(h))
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar


class Decoder(torch.nn.Module):
    """
    潜空间 -> FC -> reshape -> ConvTranspose层 -> 16x16
    结构设计: 16 -> 64*2*2 (256) -> 4x4x48 -> 8x8x24 -> 16x16x1 (与 train_vae.py 100% 同步)
    """
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.fc = torch.nn.Linear(latent_dim, 64 * 2 * 2)
        self.deconv1 = torch.nn.ConvTranspose2d(64, 48, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = torch.nn.ConvTranspose2d(48, 24, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = torch.nn.ConvTranspose2d(24, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, z):
        x = F.relu(self.fc(z))
        x = x.view(-1, 64, 2, 2)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x))
        return x


class VAE(torch.nn.Module):
    """
    完整 VAE 模型 (与 train_vae.py 100% 保持一致)
    """
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        # 辅助分类头：在训练时引导潜空间对 0-9 形成清晰聚类与超球分离
        self.classifier = torch.nn.Linear(latent_dim, 10)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decoder(z)
        pred = self.classifier(mu)
        return recon, mu, logvar, pred

    def decode(self, z):
        return self.decoder(z)


# ---- Demo 主逻辑 ----
class ToyDemo:
    # 显示配置 (保持原版设计)
    PIXEL_SIZE = 24  # 每个"LED"像素的屏幕大小
    GRID_GAP = 1     # 像素间间距
    WINDOW_W = IMAGE_SIZE * (PIXEL_SIZE + GRID_GAP) + 100  # 右边留文字区
    WINDOW_H = IMAGE_SIZE * (PIXEL_SIZE + GRID_GAP) + 60   # 底部留提示区

    # 动画与采样配置 (与单片机 mcu_code_main_modify 保持 100% 同步)
    TRANSITION_FRAMES = 45   # 45 帧 @ 60fps = 0.75 秒 (0.5~1.0 秒最佳观察区间)
    DIVERSITY_SCALE = 0.65   # 潜空间采样多样性扰动系数 (与 MCU 代码一致)

    # 颜色: 模拟暖色复古 LED
    LED_ON = (255, 180, 50)
    LED_OFF = (20, 20, 25)
    BG_COLOR = (10, 10, 15)

    # 键盘按键映射表（解决 SDL2/Pygame2 中小键盘 K_KP0..K_KP9 键码非连续递增的问题）
    KEY_MAP = {
        pygame.K_0: 0, pygame.K_1: 1, pygame.K_2: 2, pygame.K_3: 3, pygame.K_4: 4,
        pygame.K_5: 5, pygame.K_6: 6, pygame.K_7: 7, pygame.K_8: 8, pygame.K_9: 9,
        pygame.K_KP0: 0, pygame.K_KP1: 1, pygame.K_KP2: 2, pygame.K_KP3: 3, pygame.K_KP4: 4,
        pygame.K_KP5: 5, pygame.K_KP6: 6, pygame.K_KP7: 7, pygame.K_KP8: 8, pygame.K_KP9: 9,
    }

    def __init__(self, model_path='vae_best.pth'):
        pygame.init()
        self.screen = pygame.display.set_mode((self.WINDOW_W, self.WINDOW_H))
        pygame.display.set_caption("MNIST VAE 数字玩具 Demo (MCU模拟版)")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 14)
        self.big_font = pygame.font.Font(None, 28)

        # 加载模型
        print("加载模型...")
        self.device = torch.device("cpu")
        self.vae = VAE().to(self.device)
        self.vae.load_state_dict(torch.load(model_path, map_location=self.device))
        self.vae.eval()
        print("模型加载完成!")

        # 加载与单片机固件相同的潜空间分布参数 (均值锚点 mu 与离散度 std)
        self.anchor_means, self.anchor_stds = self._load_latent_distributions()

        # 状态
        self.current_digit = None
        self.current_z = torch.zeros(1, LATENT_DIM)
        self.target_z = torch.zeros(1, LATENT_DIM)
        self.transition_progress = 1.0  # 1.0 = 过渡完成
        self.frame_count = 0

    def _load_latent_distributions(self):
        """
        加载潜空间分布参数。
        优先直接读取 mcu_code_main_modify 中的头文件，确保 PC 模拟与 MCU 完全一致；
        若不存在则自动通过模型与测试集直接计算。
        """
        h_paths = [
            'mcu_code_main_modify/mcu_code_decoder_weights.h',
            'mcu_code_decoder_weights.h',
            'mcu_code_main/mcu_code_decoder_weights.h'
        ]
        for p in h_paths:
            if os.path.exists(p):
                try:
                    with open(p, 'r', encoding='utf-8') as f:
                        text = f.read()

                    def parse_arr(name):
                        m = re.search(name + r'\[10\]\[\d+\]\s*=\s*\{([^;]+)\};', text, re.DOTALL)
                        if not m: return None
                        rows = re.findall(r'\{([^}]+)\}', m.group(1))
                        return np.array([
                            [float(x.strip()) / 256.0 for x in r.split(',') if x.strip()]
                            for r in rows
                        ], dtype=np.float32)

                    means = parse_arr('anchor_points')
                    stds = parse_arr('anchor_stds')
                    if means is not None and stds is not None:
                        print(f"成功从单片机头文件 {p} 加载分布参数!")
                        return means, stds
                except Exception as e:
                    print(f"读取 {p} 失败: {e}，将尝试计算...")

        # 回退逻辑：从测试集动态计算
        print("从测试集动态统计潜空间分布...")
        from torchvision import datasets, transforms
        from torch.utils.data import DataLoader
        dataset = datasets.MNIST(
            root='./data', train=False, download=True,
            transform=transforms.Compose([
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor()
            ])
        )
        loader = DataLoader(dataset, batch_size=256, shuffle=False)
        z_by_digit = {i: [] for i in range(10)}
        with torch.no_grad():
            for data, labels in loader:
                data = data.to(self.device)
                mu, _ = self.vae.encoder(data)
                for i in range(len(labels)):
                    d = labels[i].item()
                    z_by_digit[d].append(mu[i].numpy())

        means = np.array([np.array(z_by_digit[d]).mean(axis=0) for d in range(10)], dtype=np.float32)
        stds = np.array([np.array(z_by_digit[d]).std(axis=0) for d in range(10)], dtype=np.float32)
        print("潜空间分布统计完成!")
        return means, stds

    def sample_z_for_digit(self, digit):
        """
        模拟单片机端的 vae_sample_digit 函数：
        利用潜空间多维高斯分布与有界截断，进行真实随机采样
        z = mean + diversity_scale * std * eps
        """
        mean = self.anchor_means[digit]
        std = self.anchor_stds[digit]

        # 截断正态分布（[-2.0, 2.0] 以内），保证字形丰富多变且 100% 可辨
        eps = np.clip(np.random.randn(LATENT_DIM), -2.0, 2.0).astype(np.float32)
        z_sample = (mean + self.DIVERSITY_SCALE * std * eps).astype(np.float32)
        return torch.from_numpy(z_sample.reshape(1, -1)).to(self.device)

    def on_key_press(self, digit):
        """处理按键事件（单片机 trigger_digit 的镜像实现）"""
        # 1. 真实潜空间随机采样新字形
        new_z = self.sample_z_for_digit(digit)

        # 2. 始终以当前正在显示的姿态为起点开启新过渡（即使连续按相同数字，也会缓缓变幻成新笔迹）
        self.current_z = self._get_interpolated_z()
        self.target_z = new_z
        self.transition_progress = 0.0
        self.current_digit = digit
        print(f"按下数字: {digit} (从潜空间采样新变体)")

    def _get_interpolated_z(self):
        """获取当前插值位置的潜向量 (Smoothstep 缓动曲线)"""
        t = self.transition_progress
        t_smooth = t * t * (3 - 2 * t)  # smoothstep: 3*t^2 - 2*t^3
        z = self.current_z * (1 - t_smooth) + self.target_z * t_smooth
        return z

    def decode_z(self, z):
        """潜向量 → 16x16 图像"""
        with torch.no_grad():
            img = self.vae.decoder(z)
        return img.squeeze().numpy()

    def draw_led_matrix(self, img):
        """绘制 16x16 LED矩阵效果"""
        offset_x = 20
        offset_y = 20

        for row in range(IMAGE_SIZE):
            for col in range(IMAGE_SIZE):
                brightness = float(img[row, col])

                x = offset_x + col * (self.PIXEL_SIZE + self.GRID_GAP)
                y = offset_y + row * (self.PIXEL_SIZE + self.GRID_GAP)

                # 根据亮度混合颜色
                r = int(self.LED_OFF[0] + (self.LED_ON[0] - self.LED_OFF[0]) * brightness)
                g = int(self.LED_OFF[1] + (self.LED_ON[1] - self.LED_OFF[1]) * brightness)
                b = int(self.LED_OFF[2] + (self.LED_ON[2] - self.LED_OFF[2]) * brightness)

                color = (
                    min(255, max(0, r)),
                    min(255, max(0, g)),
                    min(255, max(0, b))
                )

                # 画圆角矩形模拟LED
                rect = pygame.Rect(x, y, self.PIXEL_SIZE, self.PIXEL_SIZE)
                pygame.draw.rect(self.screen, color, rect, border_radius=3)

                # 高亮度时添加辉光效果
                if brightness > 0.6:
                    glow_alpha = int((brightness - 0.6) * 100)
                    glow_color = (
                        min(255, r + glow_alpha),
                        min(255, g + glow_alpha // 2),
                        min(255, b)
                    )
                    inner_rect = rect.inflate(-4, -4)
                    pygame.draw.rect(self.screen, glow_color, inner_rect, border_radius=2)

    def draw_ui(self):
        """绘制UI信息"""
        info_x = IMAGE_SIZE * (self.PIXEL_SIZE + self.GRID_GAP) + 30

        # 当前数字
        if self.current_digit is not None:
            text = self.big_font.render(f"Digit: {self.current_digit}", True, (200, 200, 200))
            self.screen.blit(text, (info_x, 30))

        # 过渡进度条
        bar_y = 80
        bar_w = 80
        bar_h = 10
        pygame.draw.rect(self.screen, (40, 40, 40), (info_x, bar_y, bar_w, bar_h))
        fill_w = int(bar_w * self.transition_progress)
        pygame.draw.rect(self.screen, (100, 200, 100), (info_x, bar_y, fill_w, bar_h))

        label = self.font.render(f"Transition: {self.transition_progress * 100:.0f}%", True, (150, 150, 150))
        self.screen.blit(label, (info_x, bar_y + 15))

        # 操作提示 (支持主键盘数字键与小键盘 NumPad)
        hints = [
            "Keys: 0-9 / NumPad",
            "Press same: morph",
            "ESC: exit",
            "",
            "Mode: MCU Gaussian",
            "Time: ~0.75s (45f)"
        ]
        for i, hint in enumerate(hints):
            text = self.font.render(hint, True, (100, 100, 100))
            self.screen.blit(text, (info_x, 140 + i * 18))

        # 底部状态栏
        status = self.font.render(
            f"Frame: {self.frame_count} | Z: [{self._get_interpolated_z()[0, 0].item():.2f}, ...]",
            True, (80, 80, 80)
        )
        self.screen.blit(status, (10, self.WINDOW_H - 25))

    def run(self):
        """主循环"""
        running = True
        print("\nDemo启动! 按 0-9 键或数字小键盘查看数字过渡与形态变幻效果")
        print("连续按同一个按键即可看到该数字缓缓变幻为新字形！")
        print("-" * 50)

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    # 优先通过精确键码字典匹配（支持主键盘 0-9 与小键盘 NumPad 0-9）
                    elif event.key in self.KEY_MAP:
                        self.on_key_press(self.KEY_MAP[event.key])
                    # 回退机制：根据按键输入的字符判断（兼容任意特殊小键盘与输入法状态）
                    elif event.unicode and event.unicode in "0123456789":
                        self.on_key_press(int(event.unicode))

            # 更新过渡 (0.75s 均匀推进)
            if self.transition_progress < 1.0:
                self.transition_progress += 1.0 / self.TRANSITION_FRAMES
                self.transition_progress = min(1.0, self.transition_progress)

            # 渲染
            self.screen.fill(self.BG_COLOR)

            # 解码当前插值潜向量
            z = self._get_interpolated_z()
            img = self.decode_z(z)

            self.draw_led_matrix(img)
            self.draw_ui()

            pygame.display.flip()
            self.clock.tick(60)
            self.frame_count += 1

        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    demo = ToyDemo()
    demo.run()