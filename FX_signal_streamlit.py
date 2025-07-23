import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# 페이지 설정
st.set_page_config(
    page_title="FX Signal Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 제목
st.title("🔄 FX Signal Analysis Dashboard")
st.markdown("USD/KRW 환율 데이터 기반 헤지 시그널 분석")

# 사이드바
st.sidebar.header("📊 Analysis Settings")

def get_closest_date(index, target_date):
    """주어진 인덱스에서 target_date 이전 또는 같은 가장 가까운 날짜 반환"""
    dates = index[index <= target_date]
    return dates.max() if not dates.empty else None

@st.cache_data
def load_and_process_data(uploaded_file):
    """데이터 로드 및 전처리"""
    try:
        data = pd.read_excel(uploaded_file, index_col=0)
        data_mod = data.dropna()
        data_mod = data.loc['2015-01-02':][['Spot']]

        new_idx = pd.date_range(start=data_mod.index.min(), end=data_mod.index.max(), freq='D')
        data_cal = data_mod.reindex(new_idx)
        data_cal['Spot'] = data_cal['Spot'].ffill()

        return data, data_mod, data_cal
    except Exception as e:
        st.error(f"데이터 로딩 중 오류가 발생했습니다: {e}")
        return None, None, None

def calculate_13612_momentum(data_mod, data_cal):
    """13612 Filtered Momentum 계산"""
    results = []

    progress_bar = st.progress(0)
    total_days = len(data_cal.index)

    for idx, d0 in enumerate(data_cal.index):
        progress_bar.progress((idx + 1) / total_days)

        d1 = get_closest_date(data_cal.index, d0 - pd.DateOffset(months=1))
        d3 = get_closest_date(data_cal.index, d0 - pd.DateOffset(months=3))
        d6 = get_closest_date(data_cal.index, d0 - pd.DateOffset(months=6))
        d12 = get_closest_date(data_cal.index, d0 - pd.DateOffset(months=12))

        try:
            p0 = data_mod.loc[get_closest_date(data_mod.index, d0), 'Spot']
            p1 = data_mod.loc[get_closest_date(data_mod.index, d1), 'Spot']
            p3 = data_mod.loc[get_closest_date(data_mod.index, d3), 'Spot']
            p6 = data_mod.loc[get_closest_date(data_mod.index, d6), 'Spot']
            p12 = data_mod.loc[get_closest_date(data_mod.index, d12), 'Spot']

            # 13612 모멘텀 계산
            mom = 12*p0/p1 + 4*p0/p3 + 2*p0/p6 + p0/p12 - 19

            results.append({
                'd0': d0,
                'd1': d1, 'd3': d3, 'd6': d6, 'd12': d12,
                'p0': p0, 'p1': p1, 'p3': p3, 'p6': p6, 'p12': p12,
                'mom': mom
            })
        except Exception:
            continue

    progress_bar.empty()

    df_mom = pd.DataFrame(results).set_index('d0').sort_index()
    df_mom['signal'] = 0
    df_mom['signal'][df_mom['mom'].apply(lambda x: x<0)] = 1
    df_mom['signal_5d'] = (df_mom['signal'].rolling(window=5).sum() == 5).astype(int)

    return df_mom

def plot_13612_momentum(df_mom):
    """13612 모멘텀 차트 생성"""
    df = df_mom.copy().reset_index()
    df['d0'] = pd.to_datetime(df['d0'])

    fig = go.Figure()

    # Spot Line Chart
    fig.add_trace(go.Scatter(
        x=df['d0'],
        y=df['p0'],
        mode='lines',
        name='USD/KRW Spot',
        line=dict(color='royalblue', width=2)
    ))

    # Signal ON
    df_sig1 = df[df['signal_5d'] == 1]
    fig.add_trace(go.Bar(
        x=df_sig1['d0'],
        y=[1]*len(df_sig1),
        name='Signal ON',
        marker_color='crimson',
        opacity=0.9,
        yaxis='y2'
    ))

    # Signal OFF
    df_sig0 = df[df['signal_5d'] == 0]
    fig.add_trace(go.Bar(
        x=df_sig0['d0'],
        y=[1]*len(df_sig0),
        name='Signal OFF',
        marker_color='lightgray',
        opacity=0.3,
        yaxis='y2'
    ))

    fig.update_layout(
        title='USD/KRW Spot & Hedge Signal (13612 Filtered Momentum)',
        xaxis_title='Date',
        yaxis=dict(title='USD/KRW Spot'),
        yaxis2=dict(
            title='Signal',
            overlaying='y',
            side='right',
            range=[0, 1.0],
            showgrid=False
        ),
        legend=dict(x=0.01, y=0.99),
        height=500
    )

    return fig

def calculate_macd(data_mod):
    """MACD 계산"""
    df = data_mod.copy()

    # SMA, EMA 계산
    df['SMA5'] = df['Spot'].rolling(window=5).mean()
    df['Price_to_SMA5'] = df['Spot'] / df['SMA5'] * 100
    df['EMA12'] = df['Spot'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Spot'].ewm(span=26, adjust=False).mean()
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Momentum'] = np.where(df['MACD'] > df['Signal'], 1, 0)

    # MACD 상태 계산
    momentum_state = []
    for i in range(len(df)):
        if i == 0:
            momentum_state.append(0)
        else:
            macd = df.iloc[i]['MACD']
            signal = df.iloc[i]['Signal']
            prev = momentum_state[i-1]

            if macd > signal and macd > 0:
                momentum_state.append(0)
            elif macd < signal and macd < 0:
                momentum_state.append(1)
            else:
                momentum_state.append(prev)

    df['MACD_state'] = momentum_state
    df['MACD_state_5d'] = (df['MACD_state'].rolling(window=5).sum() == 5).astype(int)

    return df

def plot_macd(df):
    """MACD 차트 생성"""
    df = df.copy().dropna(subset=['Spot', 'MACD_state_5d']).copy()
    df = df.reset_index()

    # 시그널 분리
    df_on = df[df['MACD_state_5d'] == 1]
    df_off = df[df['MACD_state_5d'] == 0]

    fig = go.Figure()

    # Spot 환율
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Spot'],
        name='USD/KRW Spot',
        mode='lines',
        line=dict(color='royalblue', width=2)
    ))

    # Signal ON
    fig.add_trace(go.Bar(
        x=df_on.index,
        y=[1] * len(df_on),
        name='Signal ON (5d)',
        marker_color='crimson',
        opacity=0.8,
        yaxis='y2'
    ))

    # Signal OFF
    fig.add_trace(go.Bar(
        x=df_off.index,
        y=[1] * len(df_off),
        name='Signal OFF',
        marker_color='lightgray',
        opacity=0.4,
        yaxis='y2'
    ))

    fig.update_layout(
        title='USD/KRW Spot with MACD 5-day Confirmed Signal',
        xaxis_title='Date',
        yaxis=dict(title='USD/KRW Spot'),
        yaxis2=dict(
            title='5-day Signal',
            overlaying='y',
            side='right',
            range=[0, 1.0],
            showgrid=False
        ),
        legend=dict(x=0.01, y=0.99),
        height=500
    )

    return fig

def calculate_basis_momentum(data):
    """Basis Momentum 계산"""
    data_mod = data.dropna()

    # 전체 달력 기준 날짜 생성
    full_dates = pd.date_range(start=data_mod.index.min(), end=data_mod.index.max(), freq='D')
    data_cal = data_mod.reindex(full_dates, method='ffill')
    data_mon = data_cal.resample('M').last()
    data_mon = data_mon.drop(index='2025-07-31').loc['2015-01-31':]

    # basis 계산
    data_mon['basis'] = np.log(data_mon['FWD1M'] / data_mon['Spot'])
    data_mon['forward_basis'] = np.log(data_mon['FWD3M'] / data_mon['FWD1M'])

    # basis momentum
    n = 3
    data_mon['basis_momentum'] = (
        data_mon['basis'].rolling(window=n).sum() -
        data_mon['forward_basis'].rolling(window=n).sum()
    )

    data_mon['signal'] = (data_mon['basis_momentum'] < 0).astype(int)
    data_mon_mod = data_mon.dropna()

    return data_mon_mod

def plot_basis_momentum(df):
    """Basis Momentum 차트 생성"""
    fig = go.Figure()

    # Spot 환율 라인
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Spot'],
        mode='lines',
        name='Spot',
        line=dict(width=2.5, color='black')
    ))

    # Signal 구간 shade
    signal_shift = df['signal'].shift(1)
    change_points = df[signal_shift != df['signal']].index.tolist()
    change_points = [df.index[0]] + change_points + [df.index[-1]]

    for i in range(len(change_points) - 1):
        x0, x1 = change_points[i], change_points[i + 1]
        current_signal = df.loc[x0, 'signal']

        fig.add_vrect(
            x0=x0, x1=x1,
            fillcolor='rgba(255, 0, 0, 0.1)' if current_signal == 1 else 'rgba(0, 200, 0, 0.08)',
            layer='below', line_width=0
        )

    fig.update_layout(
        title='현·선물 Basis Momentum',
        xaxis_title='날짜',
        yaxis_title='USD/KRW Spot',
        height=600,
        template='plotly_white',
        margin=dict(t=80, b=40, l=60, r=40),
        xaxis=dict(
            showgrid=True,
            showline=True,
            linewidth=1,
            linecolor='black',
            mirror=True
        ),
        yaxis=dict(
            range=[1000, df['Spot'].max() * 1.01],
            showgrid=True,
            zeroline=False,
            showline=True,
            linewidth=1,
            linecolor='black',
            mirror=True
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )

    return fig

# 메인 애플리케이션
def main():
    # 파일 업로드
    uploaded_file = st.sidebar.file_uploader(
        "📁 Excel 파일 업로드",
        type=['xlsx', 'xls'],
        help="usdkrw_approved_mod_v5.xlsx 형식의 파일을 업로드하세요"
    )

    if uploaded_file is not None:
        # 데이터 로드
        with st.spinner('데이터를 로딩 중입니다...'):
            data, data_mod, data_cal = load_and_process_data(uploaded_file)

        if data is not None:
            st.success("✅ 데이터가 성공적으로 로드되었습니다!")

            # 분석 방법 선택
            analysis_type = st.sidebar.selectbox(
                "🔍 분석 방법 선택",
                ["13612 Filtered Momentum", "MACD", "Basis Momentum"],
                help="분석할 시그널 방법을 선택하세요"
            )

            # 데이터 기본 정보
            st.sidebar.markdown("### 📊 데이터 정보")
            st.sidebar.write(f"**기간**: {data_mod.index.min().strftime('%Y-%m-%d')} ~ {data_mod.index.max().strftime('%Y-%m-%d')}")
            st.sidebar.write(f"**총 데이터 포인트**: {len(data_mod):,}개")

            # 탭 생성
            tab1, tab2, tab3 = st.tabs(["📈 차트 분석", "📋 데이터 테이블", "📊 통계"])

            if analysis_type == "13612 Filtered Momentum":
                with st.spinner('13612 Momentum을 계산 중입니다...'):
                    df_mom = calculate_13612_momentum(data_mod, data_cal)

                with tab1:
                    st.plotly_chart(plot_13612_momentum(df_mom), use_container_width=True)

                    # 시그널 통계
                    signal_stats = df_mom['signal_5d'].value_counts()
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📊 총 관측일", len(df_mom))
                    with col2:
                        st.metric("🔴 헤지 신호 (ON)", signal_stats.get(1, 0))
                    with col3:
                        st.metric("⚪ 비헤지 신호 (OFF)", signal_stats.get(0, 0))

                with tab2:
                    st.dataframe(df_mom.tail(100), use_container_width=True)

                with tab3:
                    st.write("### 모멘텀 분포")
                    fig_hist = go.Figure(data=[go.Histogram(x=df_mom['mom'], nbinsx=50)])
                    fig_hist.update_layout(title='13612 Momentum 분포', xaxis_title='Momentum', yaxis_title='빈도')
                    st.plotly_chart(fig_hist, use_container_width=True)

            elif analysis_type == "MACD":
                with st.spinner('MACD를 계산 중입니다...'):
                    df_macd = calculate_macd(data_mod)

                with tab1:
                    st.plotly_chart(plot_macd(df_macd), use_container_width=True)

                    # MACD 시그널 통계
                    signal_stats = df_macd['MACD_state_5d'].value_counts()
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("📊 총 관측일", len(df_macd.dropna()))
                    with col2:
                        st.metric("🔴 헤지 신호 (ON)", signal_stats.get(1, 0))
                    with col3:
                        st.metric("⚪ 비헤지 신호 (OFF)", signal_stats.get(0, 0))

                with tab2:
                    st.dataframe(df_macd[['Spot', 'MACD', 'Signal', 'MACD_state', 'MACD_state_5d']].tail(100), use_container_width=True)

                with tab3:
                    st.write("### MACD 분포")
                    fig_hist = go.Figure(data=[go.Histogram(x=df_macd['MACD'].dropna(), nbinsx=50)])
                    fig_hist.update_layout(title='MACD 분포', xaxis_title='MACD', yaxis_title='빈도')
                    st.plotly_chart(fig_hist, use_container_width=True)

            elif analysis_type == "Basis Momentum":
                if 'FWD1M' in data.columns and 'FWD3M' in data.columns:
                    with st.spinner('Basis Momentum을 계산 중입니다...'):
                        df_basis = calculate_basis_momentum(data)

                    with tab1:
                        st.plotly_chart(plot_basis_momentum(df_basis), use_container_width=True)

                        # Basis 시그널 통계
                        signal_stats = df_basis['signal'].value_counts()
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("📊 총 관측 월", len(df_basis))
                        with col2:
                            st.metric("🔴 헤지 신호 (ON)", signal_stats.get(1, 0))
                        with col3:
                            st.metric("⚪ 비헤지 신호 (OFF)", signal_stats.get(0, 0))

                    with tab2:
                        st.dataframe(df_basis[['Spot', 'FWD1M', 'FWD3M', 'basis', 'forward_basis', 'basis_momentum', 'signal']].tail(50), use_container_width=True)

                    with tab3:
                        st.write("### Basis Momentum 분포")
                        fig_hist = go.Figure(data=[go.Histogram(x=df_basis['basis_momentum'].dropna(), nbinsx=30)])
                        fig_hist.update_layout(title='Basis Momentum 분포', xaxis_title='Basis Momentum', yaxis_title='빈도')
                        st.plotly_chart(fig_hist, use_container_width=True)
                else:
                    st.error("❌ Basis Momentum 분석을 위해서는 FWD1M, FWD3M 컬럼이 필요합니다.")

    else:
        st.info("👈 사이드바에서 Excel 파일을 업로드하여 분석을 시작하세요.")

        # 샘플 데이터 구조 안내
        st.markdown("""
        ### 📋 필요한 데이터 구조
        업로드할 Excel 파일은 다음과 같은 구조여야 합니다:

        | Date | Spot | FWD1M | FWD3M |
        |------|------|-------|-------|
        | 2015-01-02 | 1100.45 | 1102.34 | 1105.67 |
        | 2015-01-05 | 1098.23 | 1100.12 | 1103.45 |
        | ... | ... | ... | ... |

        - **Date**: 날짜 (인덱스)
        - **Spot**: 현물 환율
        - **FWD1M**: 1개월 선물 환율 (Basis Momentum 분석용)
        - **FWD3M**: 3개월 선물 환율 (Basis Momentum 분석용)
        """)

if __name__ == "__main__":
    main()
