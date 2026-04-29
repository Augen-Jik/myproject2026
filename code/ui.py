import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import json
from typing import Dict, List, Tuple, Any
from eval_metrics import compute_edge_metrics
from utils_df_safe import make_arrow_safe


def _safe_stringify(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value) if isinstance(value, object) and not isinstance(value, (int, float, bool)) else value


def _safe_dataframe(df: pd.DataFrame):
    safe_df = df.copy()
    for column in safe_df.columns:
        if pd.api.types.is_object_dtype(safe_df[column].dtype) or pd.api.types.is_string_dtype(safe_df[column].dtype):
            safe_df[column] = safe_df[column].map(_safe_stringify)
    safe_df = make_arrow_safe(safe_df)
    try:
        st.dataframe(safe_df, use_container_width=True, hide_index=True)
    except Exception:
        st.markdown(safe_df.to_html(index=False, escape=True), unsafe_allow_html=True)


def render_main_header():
    """Render the main application header."""
    st.title("城市交通信号灯智能调度系统")
    st.subheader("Urban Traffic Signal Intelligent Scheduling System")

def render_sidebar_controls():
    """Render sidebar controls for the application."""
    with st.sidebar:
        st.header("控制面板")
        st.markdown("---")
        
        # Basic configuration
        demo_mode = st.checkbox("演示模式 (Demo Mode)", value=True, 
                               help="启用快速测试模式，使用简化参数")
        debug_mode = st.checkbox("调试模式 (Debug Mode)", value=False)
        
        st.markdown("---")
        
        # Algorithm selection
        algorithm = st.selectbox(
            "选择算法",
            ["DQN", "GCN-Weight", "LLM-Qwen", "LLM-R1-SFT", "LLM-R1-Raw", "LLM-Sparse", "A*路径规划", "Dijkstra最短路径"],
            index=0
        )
        
        st.markdown("---")
        
        # Experiment settings
        experiment_mode = st.radio(
            "运行模式",
            ("demo_mode", "experiment_mode"),
            format_func=lambda x: "演示模式" if x == "demo_mode" else "实验模式"
        )
        
        return {
            "demo_mode": demo_mode,
            "debug_mode": debug_mode,
            "algorithm": algorithm,
            "experiment_mode": experiment_mode
        }

def render_results_section(results_data: Dict[str, Any]):
    """Render the results section with metrics and charts."""
    st.header("📊 实验结果")
    
    if not results_data:
        st.warning("暂无结果数据")
        return
    
    # Display basic metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("总边数", results_data.get('total_edges', 'N/A'))
    with col2:
        st.metric("平均延迟", f"{results_data.get('avg_delay', 'N/A'):.2f}s")
    with col3:
        st.metric("成功率", f"{results_data.get('success_rate', 'N/A')}%")
    with col4:
        st.metric("运行时间", f"{results_data.get('execution_time', 'N/A'):.2f}s")
    
    # Show detailed results
    if 'metrics' in results_data:
        st.subheader("详细指标")
        metrics_df = pd.DataFrame([results_data['metrics']])
        _safe_dataframe(metrics_df)

def render_comparison_charts(data: List[Dict], title: str = "算法对比"):
    """Render comparison charts for multiple algorithm results."""
    if not data:
        st.warning("暂无可比较的数据")
        return
    
    st.subheader(title)
    
    # Prepare data for plotting
    df_list = []
    for idx, result in enumerate(data):
        if isinstance(result, dict):
            df_temp = pd.DataFrame([result])
            df_temp['run_id'] = idx
            df_list.append(df_temp)
    
    if df_list:
        df = pd.concat(df_list, ignore_index=True)
        
        # Create comparison chart
        fig = go.Figure()
        
        # Add traces for different metrics
        if 'avg_delay' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, 
                y=df['avg_delay'], 
                mode='lines+markers',
                name='平均延迟',
                yaxis='y'
            ))
        
        if 'success_rate' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, 
                y=df['success_rate'], 
                mode='lines+markers',
                name='成功率',
                yaxis='y2'
            ))
        
        fig.update_layout(
            title="算法性能对比",
            xaxis_title="运行序号",
            yaxis_title="平均延迟 (秒)",
            yaxis2=dict(title="成功率 (%)", overlaying='y', side='right'),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)

def display_error_details(error_info: Dict[str, Any]):
    """Display detailed error information."""
    if not error_info:
        return
    
    with st.expander("❌ 错误详情", expanded=True):
        st.json(error_info)

def display_status(status_text: str, status_type: str = "info"):
    """Display status messages with appropriate styling."""
    if status_type == "success":
        st.success(status_text)
    elif status_type == "error":
        st.error(status_text)
    elif status_type == "warning":
        st.warning(status_text)
    else:
        st.info(status_text)

def render_algorithm_specific_params(algorithm: str) -> Dict[str, Any]:
    """Render algorithm-specific parameters."""
    params = {}
    
    if algorithm.startswith("LLM-"):
        with st.expander(f"{algorithm} 参数设置", expanded=True):
            params['temperature'] = st.slider("Temperature", 0.0, 1.0, 0.7)
            params['max_tokens'] = st.slider("Max Tokens", 100, 2000, 512)
            params['top_p'] = st.slider("Top-p", 0.0, 1.0, 0.9)
    
    elif algorithm == "DQN":
        with st.expander("DQN 参数设置", expanded=True):
            params['learning_rate'] = st.slider("Learning Rate", 0.0001, 0.01, 0.001)
            params['gamma'] = st.slider("Gamma", 0.8, 0.99, 0.95)
            params['epsilon'] = st.slider("Epsilon", 0.01, 1.0, 0.1)
    
    elif algorithm in ["A*路径规划", "Dijkstra最短路径"]:
        with st.expander(f"{algorithm} 参数设置", expanded=True):
            params['weight_factor'] = st.slider("权重因子", 0.1, 3.0, 1.0)
    
    return params
