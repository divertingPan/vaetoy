"""
MNIST VAE for 16x16 LED Matrix Display
训练一个轻量VAE，用于数字平滑过渡玩具
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import json
import os
import time
import argparse

# ============================================================
# 超参数
# ============================================================
LATENT_DIM = 16  # 潜空间维度 (针对ESP32-S3升级至16维，空间分离度与多样性极佳)
IMAGE_SIZE = 16  # 输出 16x16
BATCH_SIZE = 512
EPOCHS = 60
LEARNING_RATE = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# 训练实时可视化配置
# ============================================================
VIS_SAMPLES = 8     # 每次刷新显示的样本数 (从当前batch中随机抽取, 不显示整个batch)
VIS_COLS = 4        # 每行显示几组 "输入/重建" 对比
VIS_INTERVAL = 0.3  # 窗口刷新间隔(秒), 越小越实时, 但会占用更多时间
VIS_PIXEL = 9       # 每个LED像素的屏幕尺寸(窗口超出屏幕时会自动缩小)

# ============================================================
# 数据预处理: 将MNIST 28x28 → 16x16
# ============================================================
transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)


# ============================================================
# VAE 模型定义
# ============================================================
class Encoder(nn.Module):
    """
    16x16 → Conv层 → 潜空间 (μ, log_var)
    结构设计: 16x16x1 → 8x8x24 → 4x4x48 → 2x2x64 → 16维潜空间
    """

    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 24, kernel_size=3, stride=2, padding=1)   # → 8x8x24
        self.conv2 = nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1)  # → 4x4x48
        self.conv3 = nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1)  # → 2x2x64

        self.fc_mu = nn.Linear(64 * 2 * 2, latent_dim)
        self.fc_logvar = nn.Linear(64 * 2 * 2, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # flatten (256)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar


class Decoder(nn.Module):
    """
    潜空间 → FC → reshape → ConvTranspose层 → 16x16
    结构设计: 16 → 64*2*2 (256) → 4x4x48 → 8x8x24 → 16x16x1
    在ESP32-S3上推理仅需~2.2ms, Flash占用仅~85KB
    """

    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64 * 2 * 2)

        # 2x2x64 → 4x4x48 → 8x8x24 → 16x16x1
        self.deconv1 = nn.ConvTranspose2d(64, 48, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv2 = nn.ConvTranspose2d(48, 24, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.deconv3 = nn.ConvTranspose2d(24, 1, kernel_size=3, stride=2, padding=1, output_padding=1)

    def forward(self, z):
        x = F.relu(self.fc(z))
        x = x.view(-1, 64, 2, 2)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x))  # sigmoid 输出 [0,1] 亮度
        return x


class VAE(nn.Module):
    def __init__(self, latent_dim=LATENT_DIM):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
        # 辅助分类头：在训练时引导潜空间对 0-9 形成清晰聚类与超球分离，彻底解决 4 和 9 等混淆问题
        # MCU推理时完全不调用此头，零额外开销
        self.classifier = nn.Linear(latent_dim, 10)

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
        """仅使用decoder，用于推理/插值"""
        return self.decoder(z)

# ============================================================
# 训练实时可视化窗口 (pygame)
# 每次刷新只从当前batch随机抽取少量样本, 左侧输入 / 右侧重建 对比
# 关闭窗口不会中断训练, 仅停止可视化
# 注: 窗口内文字用英文, pygame 默认字体不含中文字形
# ============================================================
class TrainingVisualizer:
    LED_OFF = (18, 18, 24)
    LED_WARM = (255, 178, 48)    # 重建图: 暖色LED
    LED_COOL = (126, 200, 255)   # 输入图: 冷色, 便于区分
    BG = (10, 10, 15)
    PANEL = (24, 24, 32)
    TEXT = (206, 206, 212)
    TEXT_DIM = (122, 122, 132)
    ACCENT = (110, 210, 130)

    MARGIN = 14
    HEADER_H = 82
    FOOTER_H = 30
    PAIR_GAP = 6     # 输入图与重建图之间的间距
    GROUP_GAP = 18   # 相邻样本组之间的间距
    LABEL_H = 22     # 每组下方的文字高度

    def __init__(self, n_samples=VIS_SAMPLES, cols=VIS_COLS,
                 interval=VIS_INTERVAL, pixel=VIS_PIXEL,
                 snapshot_dir="vis_snapshots"):
        self.n_samples = max(1, int(n_samples))
        self.cols = max(1, min(int(cols), self.n_samples))
        self.interval = max(0.02, float(interval))
        self.snapshot_dir = snapshot_dir

        self.enabled = False
        self.paused = False
        self.last_draw = 0.0
        self.loss_history = []
        self._screen = None
        self._snap_count = 0

        try:
            import pygame
        except ImportError:
            print("[可视化] 未安装 pygame, 跳过实时窗口 (pip install pygame)")
            return
        self.pygame = pygame

        try:
            pygame.init()
            pygame.display.init()
            info = pygame.display.Info()
            pixel = self._fit_pixel(pixel, info.current_w, info.current_h)
            self._layout(pixel)
            self._screen = pygame.display.set_mode((self.win_w, self.win_h))
            pygame.display.set_caption("VAE Training Monitor")
            self.font = pygame.font.Font(None, 16)
            self.font_mid = pygame.font.Font(None, 20)
            self.font_big = pygame.font.Font(None, 28)
            self.enabled = True
            print(f"[可视化] 实时窗口已启动 {self.win_w}x{self.win_h}, "
                  f"每次刷新显示 {self.n_samples} 个样本 / batch={BATCH_SIZE}")
        except Exception as e:
            print(f"[可视化] 窗口启动失败, 训练将正常继续: {e}")
            self.enabled = False

    # ---- 布局计算 ----
    def _compute_size(self, pixel):
        block = IMAGE_SIZE * pixel
        group_w = block * 2 + self.PAIR_GAP
        group_h = block + self.LABEL_H
        rows = (self.n_samples + self.cols - 1) // self.cols
        w = self.MARGIN * 2 + self.cols * group_w + (self.cols - 1) * self.GROUP_GAP
        h = self.HEADER_H + rows * group_h + self.FOOTER_H + self.MARGIN
        return w, h, block, group_w, group_h

    def _fit_pixel(self, pixel, screen_w, screen_h):
        """窗口超出屏幕时自动缩小像素尺寸"""
        while pixel > 3:
            w, h, *_ = self._compute_size(pixel)
            if w <= screen_w - 40 and h <= screen_h - 90:
                break
            pixel -= 1
        return pixel

    def _layout(self, pixel):
        self.pixel = pixel
        (self.win_w, self.win_h, self.block,
         self.group_w, self.group_h) = self._compute_size(pixel)

    # ---- 事件处理 ----
    def _pump_events(self):
        pg = self.pygame
        for event in pg.event.get():
            if event.type == pg.QUIT:
                self.close()
                return
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_ESCAPE:
                    self.close()
                elif event.key in (pg.K_p, pg.K_SPACE):
                    self.paused = not self.paused
                    print(f"[可视化] {'已暂停刷新 (按P继续)' if self.paused else '继续刷新'}")
                elif event.key == pg.K_s:
                    self._save_snapshot()

    def _save_snapshot(self):
        try:
            os.makedirs(self.snapshot_dir, exist_ok=True)
            self._snap_count += 1
            path = os.path.join(self.snapshot_dir, f"train_{self._snap_count:03d}.png")
            self.pygame.image.save(self._screen, path)
            print(f"[可视化] 截图已保存: {os.path.abspath(path)}")
        except Exception as e:
            print(f"[可视化] 截图失败: {e}")

    def close(self):
        if not self.enabled:
            return
        self.enabled = False
        try:
            self.pygame.quit()
        except Exception:
            pass
        print("[可视化] 窗口已关闭, 训练继续")

    # ---- 绘制 ----
    def _draw_led(self, img, ox, oy, warm=True):
        """把 16x16 亮度矩阵画成LED点阵"""
        pg = self.pygame
        on = self.LED_WARM if warm else self.LED_COOL
        off = self.LED_OFF
        p = self.pixel
        for r in range(IMAGE_SIZE):
            row = img[r]
            for c in range(IMAGE_SIZE):
                b = float(row[c])
                if b < 0.02:
                    color = off
                else:
                    color = (
                        min(255, int(off[0] + (on[0] - off[0]) * b)),
                        min(255, int(off[1] + (on[1] - off[1]) * b)),
                        min(255, int(off[2] + (on[2] - off[2]) * b)),
                    )
                rect = pg.Rect(ox + c * p, oy + r * p, p - 1, p - 1)
                pg.draw.rect(self._screen, color, rect,
                             border_radius=max(0, p // 4))

    def _draw_sparkline(self, rect):
        """header 里的 loss 曲线"""
        pg = self.pygame
        pg.draw.rect(self._screen, self.PANEL, rect, border_radius=3)
        hist = self.loss_history
        if len(hist) < 2:
            return
        lo, hi = min(hist), max(hist)
        span = (hi - lo) or 1.0
        n = len(hist)
        pts = []
        for i, v in enumerate(hist):
            x = rect.x + 2 + (rect.w - 4) * i / (n - 1)
            y = rect.bottom - 3 - (rect.h - 6) * (hi - v) / span
            pts.append((x, y))
        pg.draw.lines(self._screen, self.ACCENT, False, pts, 1)

    def _draw_header(self, epoch, step, total_steps, stats):
        pg = self.pygame
        s = self._screen
        pg.draw.rect(s, self.PANEL, (0, 0, self.win_w, self.HEADER_H - 10))

        prog = (step / total_steps) if total_steps else 0.0
        title = self.font_big.render(
            f"Epoch {epoch}/{EPOCHS}   Step {step}/{total_steps}", True, self.TEXT)
        s.blit(title, (self.MARGIN, 8))

        line = (f"batch loss {stats['loss']:.1f}   "
                f"recon/px {stats['recon']:.4f}   "
                f"kl/px {stats['kl']:.4f}   "
                f"{stats['samples_per_sec']:.0f} img/s")
        s.blit(self.font_mid.render(line, True, self.TEXT_DIM), (self.MARGIN, 38))

        # 进度条
        bar = pg.Rect(self.MARGIN, 62, min(220, self.win_w // 3), 6)
        pg.draw.rect(s, (44, 44, 52), bar, border_radius=3)
        pg.draw.rect(s, self.ACCENT,
                     (bar.x, bar.y, int(bar.w * prog), bar.h), border_radius=3)

        # loss 曲线
        spark_w = max(120, self.win_w - bar.right - self.MARGIN * 3 - 90)
        self._draw_sparkline(pg.Rect(bar.right + self.MARGIN + 90, 56,
                                     spark_w, self.HEADER_H - 68))
        s.blit(self.font.render("loss", True, self.TEXT_DIM),
               (bar.right + self.MARGIN + 60, 58))

    def _draw_footer(self):
        hint = ("[P] pause    [S] snapshot    close window / [ESC] stop monitoring "
                "(training continues)")
        if self.paused:
            hint = "PAUSED - press P to resume | " + hint
        color = self.ACCENT if self.paused else self.TEXT_DIM
        self._screen.blit(self.font.render(hint, True, color),
                          (self.MARGIN, self.win_h - self.FOOTER_H + 8))

    def update(self, data, recon, labels, epoch, step, total_steps, stats):
        """训练循环中每个batch调用一次, 内部按 interval 节流重绘"""
        if not self.enabled:
            return
        self._pump_events()
        if not self.enabled:
            return

        self.loss_history.append(stats["loss"])
        if len(self.loss_history) > 240:
            self.loss_history.pop(0)

        now = time.time()
        if self.paused or (now - self.last_draw) < self.interval:
            return
        self.last_draw = now

        pg = self.pygame
        s = self._screen
        s.fill(self.BG)
        self._draw_header(epoch, step, total_steps, stats)

        n = min(self.n_samples, data.size(0))
        idx = torch.randperm(data.size(0))[:n].tolist()

        x_cpu = data.detach().cpu()
        r_cpu = recon.detach().cpu().clamp(0, 1)

        for k, i in enumerate(idx):
            row, col = divmod(k, self.cols)
            gx = self.MARGIN + col * (self.group_w + self.GROUP_GAP)
            gy = self.HEADER_H + row * self.group_h

            src = x_cpu[i, 0].numpy()
            out = r_cpu[i, 0].numpy()
            self._draw_led(src, gx, gy, warm=False)
            self._draw_led(out, gx + self.block + self.PAIR_GAP, gy, warm=True)

            # 每个样本的重建误差 (BCE sum, 与训练损失同口径)
            per = F.binary_cross_entropy(r_cpu[i], x_cpu[i], reduction='sum').item()
            cap = f"label {int(labels[i])}   in | recon   bce {per:.1f}"
            s.blit(self.font.render(cap, True, self.TEXT_DIM),
                   (gx, gy + self.block + 4))

        self._draw_footer()
        pg.display.flip()

    def finalize(self):
        """训练结束后保持窗口显示最后一帧, 直到用户关闭"""
        if not self.enabled:
            return
        print("[可视化] 训练结束, 窗口保留最后一帧, 关闭窗口即可退出")
        while self.enabled:
            self._pump_events()
            self.pygame.time.wait(50)


# ============================================================
# 损失函数
# ============================================================
def vae_loss(recon_x, x, mu, logvar, pred=None, labels=None):
    # 重建损失 (BCE) + KL散度 + 辅助分类分离损失(消除4和9等相似数字混淆)
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    cls_loss = torch.tensor(0.0, device=x.device)
    if pred is not None and labels is not None:
        cls_loss = F.cross_entropy(pred, labels, reduction='sum')
    total_loss = recon_loss + kl_loss + 2.5 * cls_loss
    return total_loss, recon_loss, kl_loss, cls_loss


# ============================================================
# 训练循环
# ============================================================
def train(visualize=True, vis_samples=VIS_SAMPLES, vis_cols=VIS_COLS,
          vis_interval=VIS_INTERVAL):
    model = VAE().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    best_loss = float('inf')

    vis = TrainingVisualizer(n_samples=vis_samples, cols=vis_cols,
                             interval=vis_interval) if visualize else None
    total_steps = len(train_loader)
    step_times = []

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        total_recon = 0
        total_kl = 0

        for batch_idx, (data, labels) in enumerate(train_loader):
            t0 = time.time()
            data = data.to(DEVICE)
            labels = labels.to(DEVICE)
            optimizer.zero_grad()

            recon, mu, logvar, pred = model(data)
            loss, recon_l, kl_l, cls_l = vae_loss(recon, data, mu, logvar, pred, labels)

            loss.backward()
            optimizer.step()

            l_val = loss.item()
            r_val = recon_l.item()
            k_val = kl_l.item()

            total_loss += l_val
            total_recon += r_val
            total_kl += k_val

            step_times.append(time.time() - t0)
            if len(step_times) > 50:
                step_times.pop(0)

            # 实时可视化: 显示当前batch的输入与重建对比
            if vis is not None and vis.enabled:
                bs = data.size(0)
                dt = sum(step_times) / len(step_times)
                stats = {
                    "loss": l_val,
                    "recon": r_val / bs,
                    "kl": k_val / bs,
                    "samples_per_sec": bs / dt if dt > 0 else 0.0,
                }
                vis.update(data, recon, labels, epoch + 1,
                           batch_idx + 1, total_steps, stats)

        avg_loss = total_loss / len(train_loader.dataset)
        avg_recon = total_recon / len(train_loader.dataset)
        avg_kl = total_kl / len(train_loader.dataset)

        print(f"Epoch {epoch + 1:3d}/{EPOCHS} | "
              f"Loss: {avg_loss:.2f} | "
              f"Recon: {avg_recon:.2f} | "
              f"KL: {avg_kl:.2f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), 'vae_best.pth')

    # 保存最终模型
    torch.save(model.state_dict(), 'vae_final.pth')
    print(f"\n训练完成! 最佳Loss: {best_loss:.2f}")

    if vis is not None:
        vis.close()

    return model


# ============================================================
# 计算每个数字的潜空间锚点 (用于MCU部署)
# ============================================================
def compute_anchor_points(model):
    """
    对每个数字 0-9，计算其潜空间的代表性均值向量(锚点)和各维度标准差(std)。
    这些数据将直接烧录到MCU的Flash中，用于高保真生成和真实随机采样。
    """
    model.eval()

    latent_by_digit = {i: [] for i in range(10)}

    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to(DEVICE)
            mu, _ = model.encoder(data)
            for i in range(len(labels)):
                digit = labels[i].item()
                latent_by_digit[digit].append(mu[i].cpu().numpy())

    anchors = {}
    stds = {}
    for digit in range(10):
        arr = np.array(latent_by_digit[digit])
        anchors[digit] = arr.mean(axis=0).tolist()
        stds[digit] = arr.std(axis=0).tolist()
        print(f"数字 {digit}: 锚点均值 = {np.round(arr.mean(axis=0), 2)} | "
              f"标准差 = {np.round(arr.std(axis=0), 2)}")

    return anchors, stds


# ============================================================
# 导出模型权重为C数组 (供MCU使用)
# ============================================================
def export_decoder_to_c(model, anchors, stds, output_dirs=None):
    """
    将decoder权重、锚点均值和标准差导出为C语言头文件及JSON，
    便于ESP32-S3等MCU直接编译使用。
    """
    if output_dirs is None:
        output_dirs = ["mcu_code_main_modify"]

    decoder = model.decoder
    total_params = sum(p.numel() for p in decoder.parameters())
    total_bytes = total_params * 2  # int16

    for out_dir in output_dirs:
        os.makedirs(out_dir, exist_ok=True)

        # 1. 导出 JSON
        # with open(f"{out_dir}/anchors.json", 'w', encoding='utf-8') as f:
        #     json.dump(anchors, f, indent=2)
        # with open(f"{out_dir}/stds.json", 'w', encoding='utf-8') as f:
        #     json.dump(stds, f, indent=2)

        # 2. 导出 C 头文件
        h_name = "mcu_code_decoder_weights.h" if "mcu_code" in out_dir else "decoder_weights.h"
        header_guard = h_name.upper().replace(".", "_")
        target_path = os.path.join(out_dir, h_name)

        with open(target_path, 'w', encoding='utf-8') as f:
            f.write(f"// Auto-generated 16-Dim Enhanced VAE Decoder weights for ESP32-S3\n")
            f.write(f"// Latent dim: {LATENT_DIM}\n")
            f.write(f"// Output: {IMAGE_SIZE}x{IMAGE_SIZE}\n\n")
            f.write(f"#ifndef {header_guard}\n")
            f.write(f"#define {header_guard}\n\n")
            f.write("#include <stdint.h>\n\n")

            layer_idx = 0
            for name, param in decoder.named_parameters():
                data = param.detach().cpu().numpy().flatten()
                data_q = np.clip(data * 256, -32768, 32767).astype(np.int16)

                var_name = f"layer{layer_idx}_{name.replace('.', '_')}"
                f.write(f"// Shape: {list(param.shape)}, original range: [{data.min():.4f}, {data.max():.4f}]\n")
                f.write(f"const int16_t {var_name}[] = {{\n")

                for i in range(0, len(data_q), 16):
                    chunk = data_q[i:i + 16]
                    f.write("    " + ", ".join(f"{v:6d}" for v in chunk) + ",\n")

                f.write(f"}};  // size: {len(data_q)}\n\n")
                layer_idx += 1

            # 导出锚点均值
            f.write("// Anchor mean points in latent space (Q8.8 fixed-point)\n")
            f.write(f"const int16_t anchor_points[10][{LATENT_DIM}] = {{\n")
            for digit in range(10):
                vals = np.array(anchors[digit])
                vals_q = np.clip(vals * 256, -32768, 32767).astype(np.int16)
                f.write("    {" + ", ".join(f"{v:6d}" for v in vals_q) + f"}}, // Digit {digit}\n")
            f.write("};\n\n")

            # 导出锚点标准差
            f.write("// Latent space standard deviations for true random sampling (Q8.8 fixed-point)\n")
            f.write(f"const int16_t anchor_stds[10][{LATENT_DIM}] = {{\n")
            for digit in range(10):
                vals = np.array(stds[digit])
                vals_q = np.clip(vals * 256, -32768, 32767).astype(np.int16)
                f.write("    {" + ", ".join(f"{v:6d}" for v in vals_q) + f"}}, // Digit {digit}\n")
            f.write("};\n\n")

            f.write(f"#endif // {header_guard}\n")

        print(f"\n权重已同步导出到 {target_path}")

    print(f"Decoder参数量: {total_params}")
    print(f"Flash占用估算: ~{total_bytes / 1024:.1f} KB")


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MNIST VAE 训练 (16x16 LED矩阵玩具)")
    parser.add_argument('--no-vis', action='store_true',
                        help='关闭训练实时可视化窗口')
    parser.add_argument('--vis-samples', type=int, default=VIS_SAMPLES,
                        help=f'每次刷新显示的样本数 (默认 {VIS_SAMPLES})')
    parser.add_argument('--vis-cols', type=int, default=VIS_COLS,
                        help=f'对比图的列数 (默认 {VIS_COLS})')
    parser.add_argument('--vis-interval', type=float, default=VIS_INTERVAL,
                        help=f'窗口刷新间隔秒数 (默认 {VIS_INTERVAL})')
    parser.add_argument('--epochs', type=int, default=None,
                        help='覆盖默认训练轮数')
    args = parser.parse_args()

    if args.epochs is not None:
        EPOCHS = args.epochs

    print("=" * 60)
    print("MNIST VAE 训练 - 16x16 LED矩阵玩具")
    print("=" * 60)
    print(f"设备: {DEVICE}")
    print(f"潜空间维度: {LATENT_DIM}")
    print(f"图像尺寸: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"训练轮数: {EPOCHS}")
    print(f"实时可视化: {'关闭' if args.no_vis else '开启'}")
    print()

    # 训练
    model = train(visualize=not args.no_vis,
                  vis_samples=args.vis_samples,
                  vis_cols=args.vis_cols,
                  vis_interval=args.vis_interval)

    # 加载最佳模型
    model.load_state_dict(torch.load('vae_best.pth'))
    model.to(DEVICE)

    # 计算锚点与方差
    print("\n" + "=" * 60)
    print("计算潜空间锚点及标准差...")
    print("=" * 60)
    anchors, stds = compute_anchor_points(model)

    # 导出
    print("\n" + "=" * 60)
    print("导出MCU部署文件...")
    print("=" * 60)
    export_decoder_to_c(model, anchors, stds)