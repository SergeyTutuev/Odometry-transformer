# Odometry-transformer: Monocular Visual Odometry with TimeSformer

**Оценка 6-DoF позы камеры по последовательности монокулярных изображений с использованием пространственно-временного Transformer'а.**

---

## 📖 О проекте

Данный репозиторий содержит реализацию пайплайна **монокулярной визуальной одометрии (VO)** на основе архитектуры **TimeSformer** (Time Series Transformer). В отличие от классических геометрических методов (ORB-SLAM, VISO2), требующих ручной настройки под сценарий, и рекуррентных сетей (DeepVO), ограниченных долгосрочной памятью, мы рассматриваем визуальную одометрию как задачу **понимания видео** (Video Understanding).

Модель принимает на вход \(N_f\) последовательных кадров, разбивает каждый кадр на патчи, проецирует их в токены и через блоки **Divided Space-Time Attention** извлекает пространственно-временные признаки для прямого регресса 6 степеней свободы (DoF) относительного движения камеры.

**Ключевые особенности реализации:**
- **Архитектура:** Vision Transformer (ViT) с раздельным пространственным и временным вниманием (`attention_type='divided_space_time'`).
- **Гибкость:** Поддержка различного количества входных кадров (`window_size`) и разрешения изображений.
- **Инференс:** Скрипты для пакетного предсказания поз и восстановления абсолютной траектории с пост-обработкой скользящим окном.
- **Визуализация:** Автоматическое построение графиков «оценка vs. ground truth» в координатах X-Z.

---

## 📂 Структура репозитория

```text
Odometry-transformer/
├── checkpoints/                  # Веса моделей и артефакты обучения (Git LFS)
│   └── Exp4/
│       ├── args.pkl              # Гиперпараметры и конфигурация
│       ├── args.txt              # Человекочитаемая версия конфигурации
│       └── checkpoint_best.pth   # Веса модели (хранится в Git LFS)
├── kitti_data/                   # Загрузчик датасета KITTI
│   ├── kitti.py                  # Dataset / DataLoader
│   └── utils.py                  # Утилиты (эйлеровы углы → матрица поворота)
├── timesformer/                  # Исходный код архитектуры TimeSformer
│   ├── models/
│   │   ├── vit.py                # VisionTransformer, Block, Attention
│   │   └── vit_utils.py          # DropPath, trunc_normal_, to_2tuple
│   └── visualization/            # Утилиты TensorBoard (опционально)
├── predict_poses/                # Результаты инференса (генерируется)
│   └── <sequence>/               # Для каждой последовательности KITTI
│       ├── pred_poses.npy        # Сырые предсказания (N, window-1, 6)
│       ├── pred_poses/           # Восстановленные абсолютные позы (.txt)
│       └── plots/                # Графики траекторий (.png)
├── build_model.py                # Фабрика модели + загрузка чекпоинта
├── train.py                      # Скрипт обучения
├── predict_poses.py              # Скрипт инференса (генерация .npy)
├── plot_results.py               # Пост-обработка + построение графиков
└── tsformer-vo.jpg               # Иллюстрация пайплайна (для README)
```
## ⚙️ Установка

### 1. Клонирование репозитория

```bash
git clone https://github.com/SergeyTutuev/Odometry-transformer.git
cd Odometry-transformer
```

### 2. Зависимости

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118  # под ваш CUDA
pip install numpy matplotlib tqdm einops tensorboard
```

## 🚀 Быстрый старт
### Обучение
Все гиперпараметры (размер окна, learning rate, архитектура) задаются в блоке args и model_params внутри train.py.

```bash
python train.py
```
Данные сохраняются в checkpoints/<checkpoint_path>/:
checkpoint_best.pth — лучшая модель по val_loss
args.pkl / args.txt — конфигурация эксперимента

Примечание: по умолчанию в args указан checkpoint_path: "checkpoints/Exp4". Вы можете изменить имя эксперимента.

### Инференс
```bash
python predict_poses.py
```
Результат: predict_poses/<sequence>/pred_poses.npy — массив формы (N, window_size - 1, 6).

### Восстановление траектории и графики
```bash
python plot_results.py
```
Результат для каждой последовательности:
predict_poses/<sequence>/pred_poses/<sequence>.txt — абсолютные позы в формате KITTI
predict_poses/<sequence>/plots/pred_traj_<sequence>.png — график траектории (X-Z)
