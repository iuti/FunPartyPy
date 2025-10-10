import sys
import subprocess
import time
import signal
from pathlib import Path

def start_process(script_path: Path, log_path: Path):
    cmd = [sys.executable, str(script_path)]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = open(log_path, "ab")
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, cwd=str(script_path.parent))
    return proc, log_file

def main():
    src_dir = Path(__file__).resolve().parent  # .../human-recog-udp/src
    repo_root = src_dir.parents[1]  # .../FunPartyPy (想定)
    # 優先してルート直下のスクリプトを探す（以前の配置に対応）、見つからなければ src 下を使う
    human_candidates = [repo_root / "humanRecog.py", src_dir / "humanRecog.py"]
    pose_candidates = [repo_root / "posetest_UDP.py", src_dir / "posetest_UDP.py"]

    human_script = next((p for p in human_candidates if p.exists()), None)
    pose_script = next((p for p in pose_candidates if p.exists()), None)

    if human_script is None or pose_script is None:
        print("エラー: humanRecog.py または posetest_UDP.py が見つかりません。候補パスを確認してください。")
        print("human candidates:", human_candidates)
        print("pose candidates:", pose_candidates)
        return

    logs_dir = src_dir.parent / "logs"
    human_log = logs_dir / "humanRecog.log"
    pose_log = logs_dir / "posetest_UDP.log"

    human_proc, human_logf = start_process(human_script, human_log)
    pose_proc, pose_logf = start_process(pose_script, pose_log)

    procs = [(human_proc, human_logf), (pose_proc, pose_logf)]

    try:
        while True:
            alive = [p for p, _ in procs if p.poll() is None]
            if not alive:
                break
            time.sleep(0.5)
    except KeyboardInterrupt:
        for p, _ in procs:
            try:
                p.terminate()
            except Exception:
                pass
        time.sleep(1)
        for p, _ in procs:
            if p.poll() is None:
                try:
                    p.kill()
                except Exception:
                    pass
    finally:
        for _, f in procs:
            try:
                f.close()
            except Exception:
                pass

if __name__ == "__main__":
    main()