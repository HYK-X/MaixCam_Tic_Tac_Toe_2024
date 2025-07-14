# MaixCam-Tic-Tac-Toe-2024

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/)

这是一个基于 Sipeed Maix 系列开发板，使用计算机视觉技术实现的三子棋（井字棋）对弈机器人项目。该项目是 **2024年全国大学生电子设计竞赛（E题 - 三子棋游戏装置）** 的一个完整解决方案。

This is a Tic-Tac-Toe playing robot project based on the Sipeed Maix development board, utilizing computer vision. It serves as a complete solution for the **2024 National Undergraduate Electronic Design Contest (Problem E - Tic-Tac-Toe Game Device)** in China.

---

## 🌟 项目亮点 | Features

* **实时棋盘检测与校正 (Real-time Board Detection & Correction):** 能够自动识别摄像头画面中的棋盘，即使棋盘发生旋转或透视变形，也能准确锁定其四个角点并进行透视校正。
* **高精度棋子识别 (Accurate Piece Recognition):**
    * 使用传统计算机视觉方法（`detect_chess_pieces_internal`）和 YOLOv5 神经网络模型（`out_detector`）相结合，稳定识别棋盘内外的黑白棋子。
    * `find_qizi_out` 函数确保在游戏开始前，能够准确识别并定位场外所有备用棋子。
* **智能对弈算法 (Intelligent Game AI):** `api_new.py` 中实现了基于 **Minimax 算法** 的三子棋AI，能够计算出最佳落子策略，实现“后手不败，先手制胜”的对弈逻辑。
* **棋局变化检测 (Game State Change Detection):** `board_checker_lite.py` 模块能够比较前后两个棋局状态，准确判断出是“新增”、“移动”还是“不变”，从而实现对人类玩家作弊（移动棋子）行为的检测。
* **图形化用户界面 (GUI):** `EZGUI.py` 是一个轻量级的触摸屏GUI框架，为项目提供了多页面、多模式选择的交互界面，提升了用户体验。
* **多模式运行 (Multiple Modes):** 支持“选棋放置”、“机器先手”和“人类先手”三种模式，满足调试和对弈的多种需求。
* **硬件解耦设计 (Decoupled Design):** 核心游戏逻辑 (`api_new.py`) 与硬件控制和视觉识别代码分离，易于维护和扩展。
* **机械臂通信 (Robotic Arm Communication):** 通过 UART 串口发送精确的物理坐标指令，控制机械臂完成“取棋”和“落子”等动作。

## 📸 效果演示 | Demo

由于隐私原因暂不提供视频演示。

## 🛠️ 系统架构 | System Architecture

```

                                +---------------------------+
                                |      摄像头 (Camera)      |
                                +-------------+-------------+
                                              |
                                              | 视频流 (Video Stream)
                                              v
\+-----------------------------+ \<---------+---------------------------+
| XYZ三轴滑台 (XYZ Plotter)   |           |    Sipeed Maixcam        |
\+-----------------------------+           |  (Development Board)    |
|                             |           +---------------------------+
|   接收坐标，控制电机        |-- UART --\>| 1. 图像处理 (CV & YOLOv5)  |
| (Receives Coords,         |           | 2. 游戏逻辑 (Game Logic)   |
|  Controls Motors)         |           | 3. 用户界面 (GUI)          |
\+-----------------------------+           +---------------------------+

````

## ⚙️ 硬件与软件需求 | Requirements

### 硬件 (Hardware)
1.  **主控板:** Sipeed Maixcam
2.  **摄像头:** Maixcam 内置摄像头
3.  **显示屏:** Maixcam 内置触摸显示屏
4.  **执行机构:** XYZ三轴滑台（类似写字机），通过UART串口指令控制

### 软件 (Software)
1.  **MaixPy 固件:** 运行在 Maixcam 上的 MicroPython 环境。
2.  **Python 库:**
    * `maix` (固件内置)
    * `numpy`
    * `opencv-python` (在PC端调试时使用，MaixPy 中有内建的 `image` 模块)
3.  **神经网络模型:**
    * `model_196601.mud`: 用于检测棋盘外棋子的模型 (YOLOv5)。
    * `model_333.mud`: 用于检测棋盘内棋局状态的模型 (YOLOv5)。

## 🚀 部署与运行 | Installation & Usage

1.  **克隆仓库:**
    ```bash
    git clone [https://github.com/your-username/Maix-Vision-Tic-Tac-Toe-Robot.git](https://github.com/your-username/Maix-Vision-Tic-Tac-Toe-Robot.git)
    ```

2.  **上传文件到 MaixPy:**
    * 将项目中的所有 `.py` 文件上传到 Maix 开发板的 `/root` 目录下。
    * 根据 `main.py` 中的路径，创建相应的模型文件夹，并将 `.mud` 模型文件上传：
        * `/root/models/maixhub/196601/model_196601.mud`
        * `/root/models/maixhub/model-333.maixcam/model_333.mud`
    * 将 `EZGUI.py` 放入 `/root/GUI/` 目录。
    * 将 `board_checker_lite.py` 和 `api_new.py` 放入 `/root/moudles/` 目录。
    > **注意:** 你也可以修改 `main.py` 中的 `sys.path.append()` 和模型加载路径以适应你自己的文件结构。

3.  **硬件连接:**
    * 连接 Maixcam 的 UART TX/RX 引脚到XYZ滑台控制器的 RX/TX 引脚。请根据 `main.py` 中的 `UART_RX_PIN` 和 `UART_TX_PIN` 变量修改为你的电路板对应的引脚。

4.  **运行主程序:**
    * 通过串口终端或 MaixPy IDE 连接到你的开发板。
    * 执行主程序:
    ```python
    import sys
    sys.path.append('/root') # 确保主文件可以被导入
    import main
    ```
    或者直接在终端运行 (如果你的 MaixPy 环境支持):
    ```bash
    python /root/main.py
    ```

5.  **开始游戏:**
    * 程序运行后，触摸屏会显示主菜单。
    * 你可以选择不同的游戏模式开始游戏。机器人会自动执行棋盘和棋子检测，然后根据选择的模式开始对弈。

## 📁 代码结构 | Code Structure

````

.
├── main.py               \# 主程序: 整合所有模块，实现业务逻辑
├── api\_new.py              \# 核心API: 封装三子棋游戏规则和Minimax对弈算法
├── board\_checker\_lite.py   \# 棋局变化检测器: 用于判断棋子新增或移动
├── EZGUI.py                \# GUI框架: 提供多页面触摸交互界面
└── models/                 \# 存放YOLOv5模型文件 (需自行创建并放置)
    ├── model\_196601.mud
    └── model\_333.mud

```

## 💡 未来可改进的方向 | Future Improvements

* **自动坐标标定:** 目前机械臂的物理坐标映射 (`map_camera_to_real_coordinates`) 依赖于硬编码的标定点，可以开发一个自动化的标定程序。
* **算法优化:** Minimax 算法可以引入 Alpha-Beta 剪枝进行优化，提高决策速度。
* **视觉鲁棒性:** 增强视觉算法对复杂光照、阴影和不同背景的适应能力。
* **代码重构:** 可以将 `main.py` 中的视觉检测函数、硬件控制函数等进一步封装成类，使结构更清晰。

## 🙏 致谢 | Acknowledgements

* 感谢 **全国大学生电子设计竞赛组委会** 提供此次富有挑战性的赛题。
* 感谢 Sipeed 公司提供的 Maix 系列开发板和 MaixPy 开源社区。

## 📄 开源许可 | License

该项目遵循 [MIT License](LICENSE) 开源许可协议。
