#!/bin/bash
# ============================================================
#  start_viz.sh — Streamlit UI 启动与管理脚本
#
#  用法示例：
#    bash start_viz.sh start --port 6006 --mode demo_mode
#    bash start_viz.sh status --port 6006
#    bash start_viz.sh logs --port 66006 -n 80
#    bash start_viz.sh restart --port 6006 --mode experiment_mode
# ============================================================

set -euo pipefail

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

PROJECT_ROOT="/root/autodl-tmp"
APP_PATH="${APP_PATH:-$PROJECT_ROOT/code/app_fixed.py}"
APP_DIR="$(dirname "$APP_PATH")"
DEFAULT_PORT="${PORT:-8504}"
DEFAULT_MODE="${RUN_MODE:-demo_mode}"
LOG_DIR="$PROJECT_ROOT/logs"
STREAMLIT_HEADLESS="${STREAMLIT_HEADLESS:-true}"
STREAMLIT_ADDRESS="${STREAMLIT_ADDRESS:-0.0.0.0}"
HEALTH_PATH="/_stcore/health"

ACTION="start"
PORT="$DEFAULT_PORT"
MODE="$DEFAULT_MODE"
TAIL_LINES=40

usage() {
    cat <<EOF
用法:
  bash start_viz.sh [action] [options]

Actions:
  start       启动 UI 服务
  stop        停止指定端口上的 UI 服务
  restart     重启指定端口上的 UI 服务
  status      查看服务状态
  logs        查看日志

Options:
  --port PORT       指定端口，默认: $DEFAULT_PORT
  --mode MODE       运行模式: demo_mode / experiment_mode，默认: $DEFAULT_MODE
  --app PATH        指定 Streamlit 应用文件，默认: $APP_PATH
  -n, --lines NUM   logs 模式下显示最后 NUM 行，默认: $TAIL_LINES
  -h, --help        显示帮助

推荐命令:
  bash start_viz.sh start --port 8504 --mode demo_mode
  bash start_viz.sh restart --port 8504 --mode experiment_mode
  bash start_viz.sh status --port 8504
  bash start_viz.sh logs --port 8504 -n 100
EOF
}

print_banner() {
    echo -e "${BLUE}"
    echo "╔══════════════════════════════════════════════════════╗"
    echo "║   大模型驱动交通路径规划可视化系统                   ║"
    echo "║   Streamlit · SUMO · Folium · Plotly               ║"
    echo "╚══════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

die() {
    echo -e "${RED}❌ $*${NC}" >&2
    exit 1
}

info() {
    echo -e "${YELLOW}$*${NC}"
}

ok() {
    echo -e "${GREEN}$*${NC}"
}

resolve_python() {
    if [[ -n "${PYTHON_BIN:-}" ]]; then
        echo "$PYTHON_BIN"
        return
    fi
    if command -v python3 >/dev/null 2>&1; then
        echo "python3"
        return
    fi
    if command -v python >/dev/null 2>&1; then
        echo "python"
        return
    fi
    die "未找到可用的 Python 解释器"
}

log_path() {
    echo "$LOG_DIR/streamlit_${PORT}.log"
}

pid_file() {
    echo "$LOG_DIR/streamlit_${PORT}.pid"
}

find_pids() {
    pgrep -f "streamlit.*$(basename "$APP_PATH").*--server.port ${PORT}" 2>/dev/null || true
}

check_requirements() {
    local py_bin="$1"
    [[ -f "$APP_PATH" ]] || die "应用文件不存在：$APP_PATH"
    mkdir -p "$LOG_DIR"
    "$py_bin" - <<'PY' >/dev/null 2>&1 || exit 11
import importlib.util
mods = ["streamlit", "folium", "streamlit_folium", "sumolib", "plotly"]
missing = [m for m in mods if importlib.util.find_spec(m) is None]
if missing:
    raise SystemExit(",".join(missing))
PY
    local status=$?
    if [[ "$status" -eq 11 ]]; then
        die "缺少依赖，请安装：streamlit folium streamlit-folium sumolib plotly"
    fi
}

health_check() {
    local port="$1"
    local py_bin
    py_bin="$(resolve_python)"
    "$py_bin" - "$port" <<'PY' >/dev/null 2>&1
import socket
import sys

port = int(sys.argv[1])
s = socket.socket()
s.settimeout(1.5)
try:
    s.connect(("127.0.0.1", port))
except OSError:
    raise SystemExit(1)
finally:
    s.close()
PY
}

show_access_info() {
    local current_port="$1"
    local current_mode="$2"
    local current_log="$3"
    local host_ip=""
    if command -v hostname >/dev/null 2>&1; then
        host_ip="$(hostname -I 2>/dev/null | awk '{print $1}')"
    fi

    echo ""
    echo -e "${BLUE}══════════════════════════════════════════════════════${NC}"
    ok "🚀 UI 服务已就绪"
    echo -e "运行模式：${YELLOW}${current_mode}${NC}"
    echo -e "监听地址：${YELLOW}${STREAMLIT_ADDRESS}:${current_port}${NC}"
    echo -e "应用文件：${YELLOW}${APP_PATH}${NC}"
    echo -e "日志文件：${YELLOW}${current_log}${NC}"
    echo ""
    echo -e "本机访问：${YELLOW}http://127.0.0.1:${current_port}${NC}"
    if [[ -n "$host_ip" ]]; then
        echo -e "局域网访问：${YELLOW}http://${host_ip}:${current_port}${NC}"
    fi
    echo -e "AutoDL 访问：在 AutoDL 控制台映射端口 ${YELLOW}${current_port}${NC}"
    echo ""
    echo -e "常用命令："
    echo -e "  查看状态：${YELLOW}bash start_viz.sh status --port ${current_port}${NC}"
    echo -e "  查看日志：${YELLOW}bash start_viz.sh logs --port ${current_port} -n 100${NC}"
    echo -e "  停止服务：${YELLOW}bash start_viz.sh stop --port ${current_port}${NC}"
    echo -e "${BLUE}══════════════════════════════════════════════════════${NC}"
}

start_service() {
    local py_bin
    py_bin="$(resolve_python)"
    check_requirements "$py_bin"
    local streamlit_cmd=()
    if "$py_bin" -m streamlit --version >/dev/null 2>&1; then
        streamlit_cmd=("$py_bin" -m streamlit)
    elif command -v streamlit >/dev/null 2>&1; then
        streamlit_cmd=("streamlit")
    else
        die "未找到 streamlit，请先在当前 Python 环境安装"
    fi
    local current_log
    current_log="$(log_path)"
    local current_pid_file
    current_pid_file="$(pid_file)"
    local existing_pids
    existing_pids="$(find_pids)"

    if [[ -n "$existing_pids" ]]; then
        info "[1/4] 端口 ${PORT} 已有旧 UI 进程，先停止"
        stop_service >/dev/null
    fi

    info "[2/4] 启动 Streamlit 服务"
    (
        cd "$APP_DIR"
        STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
        RUN_MODE="$MODE" \
        nohup "${streamlit_cmd[@]}" run "$APP_PATH" \
            --server.port "$PORT" \
            --server.address "$STREAMLIT_ADDRESS" \
            --server.headless "$STREAMLIT_HEADLESS" \
            --server.enableCORS false \
            --server.enableXsrfProtection false \
            > "$current_log" 2>&1 &
        echo $! > "$current_pid_file"
    )

    local pid
    pid="$(cat "$current_pid_file")"

    info "[3/4] 等待服务变为健康状态"
    local i
    for i in $(seq 1 20); do
        if ps -p "$pid" >/dev/null 2>&1 && health_check "$PORT"; then
            ok "✅ 启动成功，PID: $pid"
            show_access_info "$PORT" "$MODE" "$current_log"
            echo ""
            echo "最近日志："
            tail -n 12 "$current_log" 2>/dev/null || true
            return
        fi
        sleep 1
    done

    echo ""
    tail -n 40 "$current_log" 2>/dev/null || true
    die "服务启动失败或健康检查未通过，请查看日志：$current_log"
}

stop_service() {
    local current_pid_file
    current_pid_file="$(pid_file)"
    local pids
    pids="$(find_pids)"

    if [[ -z "$pids" && -f "$current_pid_file" ]]; then
        pids="$(cat "$current_pid_file" 2>/dev/null || true)"
    fi

    if [[ -z "$pids" ]]; then
        ok "✅ 端口 ${PORT} 上没有发现运行中的 UI 服务"
        rm -f "$current_pid_file"
        return
    fi

    info "停止端口 ${PORT} 上的 UI 服务: $pids"
    echo "$pids" | xargs -r kill 2>/dev/null || true
    sleep 2

    local remain
    remain="$(find_pids)"
    if [[ -n "$remain" ]]; then
        info "检测到残留进程，执行强制停止"
        echo "$remain" | xargs -r kill -9 2>/dev/null || true
    fi

    rm -f "$current_pid_file"
    ok "✅ UI 服务已停止"
}

status_service() {
    local current_log
    current_log="$(log_path)"
    local pids
    pids="$(find_pids)"

    if [[ -n "$pids" ]]; then
        ok "✅ UI 服务运行中"
        echo "端口：$PORT"
        echo "PID：$pids"
        echo "日志：$current_log"
        if health_check "$PORT"; then
            ok "健康检查：通过"
        else
            info "健康检查：未通过，但进程存在"
        fi
    else
        info "UI 服务未运行"
        echo "端口：$PORT"
        echo "日志：$current_log"
    fi
}

logs_service() {
    local current_log
    current_log="$(log_path)"
    [[ -f "$current_log" ]] || die "日志文件不存在：$current_log"
    tail -n "$TAIL_LINES" "$current_log"
}

parse_args() {
    if [[ $# -gt 0 ]]; then
        case "$1" in
            start|stop|restart|status|logs)
                ACTION="$1"
                shift
                ;;
        esac
    fi

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --port)
                PORT="$2"
                shift 2
                ;;
            --mode)
                MODE="$2"
                shift 2
                ;;
            --app)
                APP_PATH="$2"
                APP_DIR="$(dirname "$APP_PATH")"
                shift 2
                ;;
            -n|--lines)
                TAIL_LINES="$2"
                shift 2
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            *)
                die "未知参数: $1"
                ;;
        esac
    done

    [[ "$MODE" == "demo_mode" || "$MODE" == "experiment_mode" ]] || die "MODE 只能是 demo_mode 或 experiment_mode"
    [[ "$PORT" =~ ^[0-9]+$ ]] || die "端口必须是数字"
}

main() {
    parse_args "$@"
    print_banner
    case "$ACTION" in
        start)
            start_service
            ;;
        stop)
            stop_service
            ;;
        restart)
            stop_service
            start_service
            ;;
        status)
            status_service
            ;;
        logs)
            logs_service
            ;;
        *)
            usage
            exit 1
            ;;
    esac
}

main "$@"
