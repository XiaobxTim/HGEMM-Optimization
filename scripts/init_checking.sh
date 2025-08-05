
echo "Initializing checking for HGEMM project... Please run this script in the root directory of the project."

# 创建data文件夹
mkdir -p data
mkdir -p data/input
mkdir -p data/output

# 创建build文件夹
mkdir -p build

# 检查python, tqdm
if ! command -v python3 &> /dev/null; then
    echo "Python3 is not installed. Please install Python3 to run the script."
    exit 1
fi

# 检查tqdm库，如果没有，提示用户安装
if ! python3 -c "import tqdm" &> /dev/null; then
    echo "tqdm is not installed. Please install tqdm to run the script."
    echo "You can install it using: pip install tqdm"
    exit 1
fi

# 检查是否有Makefile
if [ ! -f Makefile ]; then
    echo "Makefile not found. Please ensure you are in the correct directory."
    exit 1
fi

# 检查是否有src目录
if [ ! -d src ]; then
    echo "src directory not found. Please ensure you are in the correct directory."
    exit 1
fi

# 检查是否有tools目录
if [ ! -d tools ]; then
    echo "tools directory not found. Please ensure you are in the correct directory."
    exit 1
fi  

# 检查是否有scripts目录
if [ ! -d scripts ]; then
    echo "scripts directory not found. Please ensure you are in the correct directory."
    exit 1
fi  

# 检查完毕
echo "Initialization complete! All necessary directories and files are present."