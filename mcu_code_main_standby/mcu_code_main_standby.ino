/**
 * @file mcu_code_main_standby.ino
 * @brief 16x16 LED 矩阵 VAE 数字玩具 - ESP32 固件 (带 3 分钟无操作自动息屏待机与一键唤醒功能)
 * 
 * 核心升级：
 * 1. 自动待机息屏：开机状态下若 3 分钟内无任何操作（按键或串口），点阵完全熄灭，系统进入低功耗待机态。
 * 2. 两次按键行为逻辑：
 *    - 待机时按动任意按键：【第一次按键】仅唤醒屏幕，点亮并恢复待机前所展示的数字字形，不触发形变。
 *    - 唤醒后再次按动按键：【后续按键】恢复正常交互，触发潜空间高斯采样与数字平滑流动变幻。
 * 3. 边缘按键检测 (Edge-Triggered)：彻底杜绝长按误连击，按一次算一次，手感利落。
 * 4. 待机功耗优化：待机期间暂停 60FPS 神经网络推理与点阵刷新，大幅降低发热和耗电。
 */

#include <FastLED.h>
#include "mcu_code_vae_engine.h"

// ============================================================================
// 1. 硬件引脚与点阵配置
// ============================================================================
#define LED_PIN          21      // WS2812B 数据线 (DIN) 接 ESP32-S3 GPIO 21 (右侧排针 Pin 17)
#define MATRIX_WIDTH     16      // 点阵宽度
#define MATRIX_HEIGHT    16      // 点阵高度
#define NUM_LEDS         (MATRIX_WIDTH * MATRIX_HEIGHT) // 256 颗 LED

// 全局亮度上限 (0-255)。推荐 25~50，室内非常清晰且总电流 < 300mA，直接用电脑 USB 供电完全无压力！
#define MAX_BRIGHTNESS   40

// 10 个按键 (0-9) 对应的 GPIO 引脚配置
// 【针对 ESP32-S3-CAM 优化】：精选左侧排针 Pin 3 ~ Pin 12 连续 10 个引脚，完美一字排开，接线极度整洁！
// 轴体一脚接对应 GPIO，另一脚统一接板子 GND！
const uint8_t KEY_PINS[10] = {
    4,   // 按键 0 -> 左侧排针 Pin 3  (GPIO 4)
    5,   // 按键 1 -> 左侧排针 Pin 4  (GPIO 5)
    6,   // 按键 2 -> 左侧排针 Pin 5  (GPIO 6)
    7,   // 按键 3 -> 左侧排针 Pin 6  (GPIO 7)
    15,  // 按键 4 -> 左侧排针 Pin 7  (GPIO 15)
    16,  // 按键 5 -> 左侧排针 Pin 8  (GPIO 16)
    17,  // 按键 6 -> 左侧排针 Pin 9  (GPIO 17)
    18,  // 按键 7 -> 左侧排针 Pin 10 (GPIO 18)
    8,   // 按键 8 -> 左侧排针 Pin 11 (GPIO 8)
    3    // 按键 9 -> 左侧排针 Pin 12 (GPIO 3)
};

// ============================================================================
// 2. 视觉效果、动画时长与多样性配置
// ============================================================================
#define TARGET_FPS        60      // 目标帧率 (60 FPS，动画丝滑流畅)

/**
 * 过渡时间配置（帧数）：
 * 30 帧 @ 60FPS = 0.5 秒 (快速利落)
 * 45 帧 @ 60FPS = 0.75 秒 (推荐黄金时间，不太快也不太慢，非常适合观察笔画流动)
 * 60 帧 @ 60FPS = 1.0 秒 (舒缓沉浸)
 */
#define TRANSITION_FRAMES 45

/**
 * 手写多样性扰动系数 (Diversity Scale)：
 * 0.0  = 每次生成固定的标准印刷体 (固定中心锚点)
 * 0.65 = 推荐值，每次按下产生各异的笔画粗细、转角弧度与倾斜度，且 100% 确保数字清晰可辨
 * 0.85 = 更大变体范围，更具个性手写狂草风格
 */
#define DIVERSITY_SCALE   0.65f

// 颜色定义 (模拟暖橙琥珀色复古 LED 发光管质感)
const CRGB COLOR_ON  = CRGB(255, 180, 50);  // 点亮时的颜色
// const CRGB COLOR_OFF = CRGB(8, 8, 12);      // 背景熄灭底色（微弱深蓝灰色）
const CRGB COLOR_OFF = CRGB(0, 0, 0);      // 背景熄灭底色

// WS2812B 显存数组
CRGB leds[NUM_LEDS];

// ============================================================================
// 3. 待机与电源管理配置
// ============================================================================
/**
 * 待机超时时长配置（毫秒）：
 * 3 分钟 = 3 * 60 * 1000 = 180,000 毫秒
 * 【调试提示】：如果想要快速测试待机功能，可临时将这行改为 10UL * 1000UL (10秒)
 */
#define STANDBY_TIMEOUT_MS  (3UL * 60UL * 1000UL) 
// #define STANDBY_TIMEOUT_MS  (10UL * 1000UL) 

bool is_standby = false;                 // 当前系统是否处于熄灯待机状态
unsigned long last_interaction_time = 0; // 最近一次检测到用户操作的时间戳

// ============================================================================
// 4. 状态变量
// ============================================================================
float current_z[VAE_LATENT_DIM];  // 当前潜向量起始点
float target_z[VAE_LATENT_DIM];   // 目标潜向量
float interp_z[VAE_LATENT_DIM];   // 当前插值计算潜向量
float output_img[VAE_IMAGE_SIZE][VAE_IMAGE_SIZE]; // VAE 解码出的 16x16 灰度图

int current_digit = 0;
int transition_step = TRANSITION_FRAMES; // 当前过渡进度（>= TRANSITION_FRAMES 表示过渡已完成）

// 按键边缘检测与消抖状态
bool last_pin_state[10] = {HIGH, HIGH, HIGH, HIGH, HIGH, HIGH, HIGH, HIGH, HIGH, HIGH};
unsigned long last_key_time[10] = {0};
const unsigned long DEBOUNCE_DELAY = 50; // 50ms 消抖

// ============================================================================
// 5. 点阵坐标映射函数 (WS2812B 16x16 蛇形走线适配)
// ============================================================================
uint16_t XY(uint8_t x, uint8_t y) {
    if (x >= MATRIX_WIDTH || y >= MATRIX_HEIGHT) return 0;
    
    // 如果发现你的点阵显示上下颠倒或左右镜像，取消下面相应注释即可纠正：
    // y = 15 - y; // 垂直翻转
    x = 15 - x; // 水平翻转

    if (y & 0x01) {
        // 奇数行：从右往左
        return (y * MATRIX_WIDTH) + (MATRIX_WIDTH - 1 - x);
    } else {
        // 偶数行：从左往右
        return (y * MATRIX_WIDTH) + x;
    }
}

// ============================================================================
// 6. VAE 画面渲染与刷新
// ============================================================================
void render_and_show() {
    // 1. 调用 VAE Decoder 推理计算 16x16 像素矩阵
    vae_decode(interp_z, output_img);

    // 2. 将推理出的浮点亮度映射为琥珀色光晕并填充点阵
    for (int row = 0; row < MATRIX_HEIGHT; row++) {
        for (int col = 0; col < MATRIX_WIDTH; col++) {
            float brightness = output_img[row][col];
            if (brightness < 0.0f) brightness = 0.0f;
            if (brightness > 1.0f) brightness = 1.0f;

            // 线性混合颜色 (模拟辉光效果)
            uint8_t r = (uint8_t)(COLOR_OFF.r + (COLOR_ON.r - COLOR_OFF.r) * brightness);
            uint8_t g = (uint8_t)(COLOR_OFF.g + (COLOR_ON.g - COLOR_OFF.g) * brightness);
            uint8_t b = (uint8_t)(COLOR_OFF.b + (COLOR_ON.b - COLOR_OFF.b) * brightness);

            // 当高亮时略微增加外圈暖辉光
            if (brightness > 0.65f) {
                float glow = (brightness - 0.65f) * 120.0f;
                r = qadd8(r, (uint8_t)glow);
                g = qadd8(g, (uint8_t)(glow * 0.6f));
            }

            uint16_t led_idx = XY(col, row);
            leds[led_idx] = CRGB(r, g, b);
        }
    }

    // 刷新显示到硬件点阵
    FastLED.show();
}

// ============================================================================
// 7. 待机唤醒与按键触发逻辑
// ============================================================================
// 唤醒系统：点亮点阵并恢复原数字显示
void wake_up_from_standby() {
    is_standby = false;
    last_interaction_time = millis();
    
    Serial.println("\n[SYSTEM] >>> 检测到唤醒按键！系统退出待机，恢复屏幕显示。");
    Serial.println("[SYSTEM] >>> 再次按下任意按键，即可开始字形变换！");
    
    // 立即重新计算并点亮恢复原来的字形
    render_and_show();
}

// 触发新的数字采样与平滑变幻
void trigger_digit(int digit) {
    if (digit < 0 || digit > 9) return;
    
    // 1. 将当前实际渲染位置作为新的起点（即使在上一次过渡中途按下，也能无缝转向）
    for (int i = 0; i < VAE_LATENT_DIM; i++) {
        current_z[i] = interp_z[i];
    }
    
    // 2. 从潜空间的真实高斯分布中随机采样全新变体！
    vae_sample_digit(digit, target_z, DIVERSITY_SCALE);
    
    current_digit = digit;
    transition_step = 0; // 重置进度，启动 0.75 秒的平滑过渡
    
    Serial.print("[KEY PRESSED] 触发数字: ");
    Serial.print(digit);
    Serial.println(" (从潜空间随机采样新字形，启动流动变幻)");
}

// 检查按键扫描 (带精准下降沿检测与消抖)
void check_buttons() {
    unsigned long now = millis();
    for (int i = 0; i < 10; i++) {
        int reading = digitalRead(KEY_PINS[i]);

        // 检测下降沿：按键从未按下(HIGH)变为按下(LOW)的瞬间
        if (reading == LOW && last_pin_state[i] == HIGH) {
            if (now - last_key_time[i] > DEBOUNCE_DELAY) {
                last_key_time[i] = now;
                last_pin_state[i] = LOW; // 锁定当前按键为按下状态

                if (is_standby) {
                    // 【关键分支 1】：当前若处于待机状态，这次按键仅负责“唤醒屏幕”，不触发变形
                    wake_up_from_standby();
                } else {
                    // 【关键分支 2】：当前已是唤醒状态，正常触发数字过渡变幻
                    last_interaction_time = now;
                    trigger_digit(i);
                }
            }
        } 
        // 检测上升沿：按键松开恢复为 HIGH
        else if (reading == HIGH && last_pin_state[i] == LOW) {
            if (now - last_key_time[i] > DEBOUNCE_DELAY) {
                last_key_time[i] = now;
                last_pin_state[i] = HIGH; // 允许下一次按下触发
            }
        }
    }
}

// 检查 USB 串口指令（方便免接线快速验证）
void check_serial() {
    while (Serial.available() > 0) {
        char ch = Serial.read();
        if (ch >= '0' && ch <= '9') {
            int digit = ch - '0';
            if (is_standby) {
                wake_up_from_standby();
            } else {
                last_interaction_time = millis();
                Serial.print("[SERIAL CMD] 收到指令: ");
                Serial.println(digit);
                trigger_digit(digit);
            }
        }
    }
}

// ============================================================================
// 8. Arduino setup() 与 loop()
// ============================================================================
void setup() {
    Serial.begin(115200);
    // 兼容 ESP32-S3 原生 USB CDC (OTG接口)，等待串口监视器握手 (超时 1.5 秒后继续，不阻塞离线启动)
    unsigned long serial_wait_start = millis();
    while (!Serial && (millis() - serial_wait_start < 1500));
    delay(200);

    Serial.println("\n=======================================================");
    Serial.println("  16x16 LED VAE 玩具固件 (带 3 分钟待机与按键唤醒版)");
    Serial.println("=======================================================");

    // 初始化硬件随机数种子
    #if defined(ESP32)
        randomSeed(esp_random());
        srand(esp_random());
    #else
        randomSeed(analogRead(0));
        srand(analogRead(0));
    #endif

    // 初始化 WS2812B 点阵
    FastLED.addLeds<WS2812B, LED_PIN, GRB>(leds, NUM_LEDS);
    FastLED.setBrightness(MAX_BRIGHTNESS);
    FastLED.clear(true);

    // 初始化按键引脚（开启内部弱上拉，按键另一端直接连 GND）
    for (int i = 0; i < 10; i++) {
        pinMode(KEY_PINS[i], INPUT_PULLUP);
    }

    // 初始化 VAE 神经网络推理引擎
    vae_init();

    // 默认展示数字 0 的一个随机字形
    vae_sample_digit(0, current_z, DIVERSITY_SCALE);
    for (int i = 0; i < VAE_LATENT_DIM; i++) {
        target_z[i] = current_z[i];
        interp_z[i] = current_z[i];
    }
    transition_step = TRANSITION_FRAMES;

    // 初次点亮显示
    render_and_show();

    // 记录初始启动时间戳
    last_interaction_time = millis();
    is_standby = false;

    Serial.println("系统就绪！");
    Serial.println("- 正常状态：按任意数字键，字形将流动变幻。");
    Serial.println("- 待机规则：3 分钟无操作自动息屏灭灯进入待机。");
    Serial.println("- 唤醒规则：待机时按动第 1 次按键唤醒屏幕恢复显示，再次按动开始字形变换。");
}

void loop() {
    // 1. 输入事件扫描
    check_buttons();
    check_serial();

    // 2. 检查 3 分钟超时待机判断
    if (!is_standby) {
        if (millis() - last_interaction_time >= STANDBY_TIMEOUT_MS) {
            is_standby = true;
            Serial.println("\n[SYSTEM] >>> 超过 3 分钟无操作，点阵熄灭进入待机模式。");
            Serial.println("[SYSTEM] >>> 功耗已降低。按任意按键即可唤醒屏幕！");
            
            // 熄灭所有点阵 LED
            FastLED.clear(true);
        }
    }

    // 3. 待机模式保护：若处于待机状态，跳过高频 VAE 解码计算与刷新，省电且防发热
    if (is_standby) {
        delay(20); // 待机期间以 50Hz 轮询按键，CPU 负载趋近于 0
        return;
    }

    // 4. 正常运行模式：60 FPS 平滑过渡渲染
    unsigned long frame_start = millis();

    // 潜空间插值计算 (Smoothstep 缓动曲线)
    if (transition_step < TRANSITION_FRAMES) {
        transition_step++;
        float t = (float)transition_step / (float)TRANSITION_FRAMES;
        // smoothstep: 3*t^2 - 2*t^3
        float smooth_t = t * t * (3.0f - 2.0f * t);

        for (int i = 0; i < VAE_LATENT_DIM; i++) {
            interp_z[i] = current_z[i] * (1.0f - smooth_t) + target_z[i] * smooth_t;
        }
    }

    // 渲染输出图像到点阵
    render_and_show();

    // 5. 帧率控制 (保持稳定的 60 FPS)
    unsigned long elapsed = millis() - frame_start;
    const unsigned long frame_time = 1000 / TARGET_FPS;
    if (elapsed < frame_time) {
        delay(frame_time - elapsed);
    }
}

