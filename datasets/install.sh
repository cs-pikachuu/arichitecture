# 1) 变量（按你刚才解包出来的目录名）
BASE="$HOME/nvml-570.169"
EXTRACT_DIR=$(ls -d $BASE/extract_* | sort | tail -n1)  # 取最新解包目录
DEST="$BASE/lib"

# 2) 确保我们用的是 64 位 NVML（排除 32/ 目录）
LIB64=$(find "$EXTRACT_DIR" -path '*/32/*' -prune -o -type f -name 'libnvidia-ml.so.570.169' -print | head -n1)
if [ -z "$LIB64" ]; then
  echo "[ERROR] 找不到 64 位 libnvidia-ml.so.570.169，请把上面命令的 EXTRACT_DIR 换成实际路径后重试。"
  exit 1
fi

# 3) 覆盖成 64 位版本 & 建立 .so.1 的软链
mkdir -p "$DEST"
cp -v "$LIB64" "$DEST/libnvidia-ml.so.570.169"

cd "$DEST"
# 建立/更新 libnvidia-ml.so.1 指向刚才那份 64 位库
ln -sfn libnvidia-ml.so.570.169 libnvidia-ml.so.1
# 可选：再给一个无后缀名的链接（有些程序会找这个）
ln -sfn libnvidia-ml.so.570.169 libnvidia-ml.so
cd - >/dev/null

# 4) 验证这份库是 64 位
file "$DEST/libnvidia-ml.so.570.169"

# 5) 用这份 570.169 的 NVML 运行 nvidia-smi（强制只在这条命令生效）
LD_LIBRARY_PATH="$DEST:$LD_LIBRARY_PATH" LD_DEBUG=libs nvidia-smi 2>&1 | grep -E 'libnvidia-ml\.so' || true
LD_LIBRARY_PATH="$DEST:$LD_LIBRARY_PATH" nvidia-smi

