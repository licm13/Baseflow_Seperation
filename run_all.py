# run_all.py
import subprocess, sys, time, pathlib

# 可改成你的脚本相对路径或绝对路径
SCRIPTS = ["batch_run_06_21.py", "batch_run_60_16.py", "batch_run_60_99.py"]

PY = sys.executable              # 当前解释器
WORKDIR = pathlib.Path(__file__).parent  # 以当前文件夹为工作目录

def run_once():
    for s in SCRIPTS:
        print(f"\n=== Running {s} ===")
        # 捕获输出到终端；出错会抛异常
        subprocess.run([PY, s], cwd=WORKDIR, check=True)

if __name__ == "__main__":
    # 只跑一遍：
    # run_once()

    # 循环轮流跑（例如每轮间隔60秒）：
    while True:
        try:
            run_once()
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] {e}. 停止或记录后再决定是否继续。")
            # break   # 想遇错停就解开这行
        time.sleep(60)  # 每轮间隔 60 秒，可按需修改
