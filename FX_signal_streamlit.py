import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from collections import defaultdict
import io

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

def calculate_basis_momentum(data):
    """Basis Momentum 계산"""
    data_mod = data.dropna()

    # 전체 달력 기준 날짜 생성
    full_dates = pd.date_range(start=data_mod.index.min(), end=data_mod.index.max(), freq='D')
    data_cal = data_mod.reindex(full_dates, method='ffill')
    data_mon = data_cal.resample('M').last()

    # 인덱스에서 2025-07-31이 있으면 제거
    if '2025-07-31' in data_mon.index:
        data_mon = data_mon.drop(index='2025-07-31')
    data_mon = data_mon.loc['2015-01-31':]

    # basis 계산
    data_mon['basis'] = np.log(data_mon['FWD1M'] / data_mon['Spot'])
    data_mon['forward_basis'] = np.log(data_mon['FWD2M'] / data_mon['FWD1M'])

    # basis momentum
    n = 3
    data_mon['basis_momentum'] = (
        data_mon['basis'].rolling(window=n).sum() -
        data_mon['forward_basis'].rolling(window=n).sum()
    )

    data_mon['signal'] = (data_mon['basis_momentum'] < 0).astype(int)
    data_mon_mod = data_mon.dropna()

    return data_mon_mod

def create_composite_signal(df_mom, df_macd, df_basis):
    """3단계 신호를 합성하여 누적 막대그래프용 데이터 생성"""
    # 공통 날짜 범위 찾기
    common_start = max(df_mom.index.min(), df_macd.index.min())
    common_end = min(df_mom.index.max(), df_macd.index.max())

    # 일간 데이터로 정렬
    df_composite = pd.DataFrame(index=pd.date_range(common_start, common_end, freq='D'))

    # 13612 시그널 (일간)
    df_composite['signal_13612'] = df_mom.reindex(df_composite.index)['signal_5d'].fillna(method='ffill')

    # MACD 시그널 (일간)
    df_composite['signal_macd'] = df_macd.reindex(df_composite.index)['MACD_state_5d'].fillna(method='ffill')

    # Basis 시그널 (월간을 일간으로 확장)
    df_composite['signal_basis'] = 0
    for date in df_composite.index:
        # 해당 날짜의 월말 찾기
        month_end = date + pd.offsets.MonthEnd(0)
        if month_end in df_basis.index:
            df_composite.loc[date, 'signal_basis'] = df_basis.loc[month_end, 'signal']
        else:
            # 가장 가까운 이전 월말 찾기
            prev_month_ends = df_basis.index[df_basis.index <= month_end]
            if len(prev_month_ends) > 0:
                df_composite.loc[date, 'signal_basis'] = df_basis.loc[prev_month_ends[-1], 'signal']

    # 환율 데이터 추가
    df_composite['spot'] = df_mom.reindex(df_composite.index)['p0'].fillna(method='ffill')

    # 누적 시그널 계산 (3단계)
    df_composite['composite_signal'] = (
        df_composite['signal_13612'].fillna(0) +
        df_composite['signal_macd'].fillna(0) +
        df_composite['signal_basis'].fillna(0)
    )

    return df_composite.dropna()

def plot_composite_signal(df_composite):
    """합성 시그널 누적 막대그래프"""
    df = df_composite.copy()

    fig = go.Figure()

    # 환율 라인
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['spot'],
        mode='lines',
        name='USD/KRW Spot',
        line=dict(color='royalblue', width=2)
    ))

    # 누적 막대그래프 - 3단계별 색상
    colors = ['lightgray', '#ffcccc', '#ff6666', '#cc0000']  # 0, 1, 2, 3 시그널별 색상

    for signal_level in [0, 1, 2, 3]:
        df_level = df[df['composite_signal'] == signal_level]
        if len(df_level) > 0:
            fig.add_trace(go.Bar(
                x=df_level.index,
                y=[1] * len(df_level),
                name=f'Signal Level {signal_level}',
                marker_color=colors[signal_level],
                opacity=0.7,
                yaxis='y2'
            ))

    fig.update_layout(
        title='USD/KRW Spot with Composite Signal (3-Stage Cumulative)',
        xaxis_title='Date',
        yaxis=dict(title='USD/KRW Spot'),
        yaxis2=dict(
            title='Signal Level',
            overlaying='y',
            side='right',
            range=[0, 1.0],
            showgrid=False
        ),
        legend=dict(x=0.01, y=0.99),
        height=600
    )

    return fig

def plot_individual_signal(df, signal_col, title):
    """개별 시그널 차트"""
    fig = go.Figure()

    # 환율 라인
    spot_col = 'p0' if 'p0' in df.columns else 'Spot'
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df[spot_col],
        mode='lines',
        name='USD/KRW Spot',
        line=dict(color='royalblue', width=2)
    ))

    # Signal ON/OFF
    df_on = df[df[signal_col] == 1]
    df_off = df[df[signal_col] == 0]

    fig.add_trace(go.Bar(
        x=df_on.index,
        y=[1] * len(df_on),
        name='Signal ON',
        marker_color='crimson',
        opacity=0.8,
        yaxis='y2'
    ))

    fig.add_trace(go.Bar(
        x=df_off.index,
        y=[1] * len(df_off),
        name='Signal OFF',
        marker_color='lightgray',
        opacity=0.4,
        yaxis='y2'
    ))

    fig.update_layout(
        title=title,
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

def plot_basis_momentum_individual(df):
    """Basis Momentum 개별 차트 - 다른 차트들과 동일한 스타일로 통일"""
    fig = go.Figure()

    # 환율 라인 - 다른 차트들과 동일한 색상으로
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['Spot'],
        mode='lines',
        name='USD/KRW Spot',
        line=dict(color='royalblue', width=2)
    ))

    # Signal ON/OFF - 다른 차트들과 동일한 스타일로
    df_on = df[df['signal'] == 1]
    df_off = df[df['signal'] == 0]

    fig.add_trace(go.Bar(
        x=df_on.index,
        y=[1] * len(df_on),
        name='Signal ON',
        marker_color='crimson',
        opacity=0.8,
        yaxis='y2'
    ))

    fig.add_trace(go.Bar(
        x=df_off.index,
        y=[1] * len(df_off),
        name='Signal OFF',
        marker_color='lightgray',
        opacity=0.4,
        yaxis='y2'
    ))

    fig.update_layout(
        title='USD/KRW Spot with Basis Momentum Signal',
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

def display_recent_signal_status(df, signal_col, signal_name):
    """최근 시그널 상태 표시"""
    if len(df) > 0:
        latest_date = df.index[-1]
        latest_signal = df[signal_col].iloc[-1]

        # 날짜 포맷 (timestamp 제거)
        if hasattr(latest_date, 'strftime'):
            formatted_date = latest_date.strftime('%Y-%m-%d')
        else:
            formatted_date = str(latest_date).split(' ')[0]

        signal_status = "🔴 ON" if latest_signal == 1 else "⚪ OFF"
        signal_color = "red" if latest_signal == 1 else "gray"

        st.markdown(f"""
        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
            <h4>📅 최근 {signal_name} 시그널 상태</h4>
            <p><strong>날짜:</strong> {formatted_date}</p>
            <p><strong>시그널:</strong> <span style="color: {signal_color}; font-weight: bold;">{signal_status}</span></p>
        </div>
        """, unsafe_allow_html=True)

def format_dataframe_dates(df):
    """데이터프레임의 날짜 컬럼 포맷팅 (timestamp 제거)"""
    df_formatted = df.copy()

    # 인덱스가 날짜인 경우
    if hasattr(df_formatted.index, 'strftime'):
        df_formatted.index = df_formatted.index.strftime('%Y-%m-%d')

    # 날짜 컬럼들 포맷팅
    date_columns = ['d1', 'd3', 'd6', 'd12']
    for col in date_columns:
        if col in df_formatted.columns:
            if hasattr(df_formatted[col], 'dt'):
                df_formatted[col] = df_formatted[col].dt.strftime('%Y-%m-%d')

    return df_formatted

# 메인 애플리케이션
def main():
    # 파일 업로드
    uploaded_file = st.sidebar.file_uploader(
        "📁 Excel 파일 업로드",
        type=['xlsx', 'xls'],
        help="usdkrw_app_mod_v6.xlsx 형식의 파일을 업로드하세요"
    )

    if uploaded_file is not None:
        # 데이터 로드
        with st.spinner('데이터를 로딩 중입니다...'):
            data, data_mod, data_cal = load_and_process_data(uploaded_file)

        if data is not None:
            st.success("✅ 데이터가 성공적으로 로드되었습니다!")

            # 데이터 기본 정보
            st.sidebar.markdown("### 📊 데이터 정보")
            st.sidebar.write(f"**기간**: {data_mod.index.min().strftime('%Y-%m-%d')} ~ {data_mod.index.max().strftime('%Y-%m-%d')}")
            st.sidebar.write(f"**총 데이터 포인트**: {len(data_mod):,}개")

            # 시그널 계산
            with st.spinner('시그널을 계산 중입니다...'):
                df_mom = calculate_13612_momentum(data_mod, data_cal)
                df_macd = calculate_macd(data_mod)

                # Basis momentum은 FWD 데이터가 있을 때만
                df_basis = None
                if 'FWD1M' in data.columns and 'FWD2M' in data.columns:
                    df_basis = calculate_basis_momentum(data)

            # 탭 생성 (4개 탭)
            if df_basis is not None:
                tab1, tab2, tab3, tab4 = st.tabs(["📊 합성 시그널", "📈 13612 Momentum", "📉 MACD", "📋 Basis Momentum"])

                # 합성 시그널 생성
                df_composite = create_composite_signal(df_mom, df_macd, df_basis)

                with tab1:
                    # 최근 합성 시그널 상태
                    if len(df_composite) > 0:
                        latest_date = df_composite.index[-1]
                        latest_composite = df_composite['composite_signal'].iloc[-1]
                        formatted_date = latest_date.strftime('%Y-%m-%d')

                        st.markdown(f"""
                        <div style="background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
                            <h4>📅 최근 합성 시그널 상태</h4>
                            <p><strong>날짜:</strong> {formatted_date}</p>
                            <p><strong>합성 시그널:</strong> <span style="color: {'red' if latest_composite >= 2 else 'orange' if latest_composite == 1 else 'gray'}; font-weight: bold;">Level {int(latest_composite)}/3</span></p>
                            <p><strong>13612:</strong> {'🔴' if df_composite['signal_13612'].iloc[-1] else '⚪'}
                            <strong>MACD:</strong> {'🔴' if df_composite['signal_macd'].iloc[-1] else '⚪'}
                            <strong>Basis:</strong> {'🔴' if df_composite['signal_basis'].iloc[-1] else '⚪'}</p>
                        </div>
                        """, unsafe_allow_html=True)

                    st.plotly_chart(plot_composite_signal(df_composite), use_container_width=True)

                    # 합성 시그널 통계
                    signal_dist = df_composite['composite_signal'].value_counts().sort_index()
                    cols = st.columns(4)
                    for i, (level, count) in enumerate(signal_dist.items()):
                        with cols[i]:
                            st.metric(f"Level {int(level)}", count)

            else:
                tab1, tab2, tab3 = st.tabs(["📈 13612 Momentum", "📉 MACD", "❌ Basis (데이터 없음)"])

            # 13612 Momentum 탭
            tab_idx = 2 if df_basis is not None else 1
            with (tab2 if df_basis is not None else tab1):
                display_recent_signal_status(df_mom, 'signal_5d', '13612 Momentum')
                st.plotly_chart(plot_individual_signal(df_mom, 'signal_5d', 'USD/KRW Spot & 13612 Filtered Momentum Signal'), use_container_width=True)

                # 데이터 테이블 (내림차순, 날짜 포맷팅)
                st.subheader("📋 최근 데이터")
                df_display = format_dataframe_dates(df_mom.tail(50).iloc[::-1])  # 내림차순
                st.dataframe(df_display, use_container_width=True)

            # MACD 탭
            with (tab3 if df_basis is not None else tab2):
                display_recent_signal_status(df_macd, 'MACD_state_5d', 'MACD')
                st.plotly_chart(plot_individual_signal(df_macd, 'MACD_state_5d', 'USD/KRW Spot with MACD 5-day Confirmed Signal'), use_container_width=True)

                # 데이터 테이블 (내림차순)
                st.subheader("📋 최근 데이터")
                df_macd_display = df_macd[['Spot', 'MACD', 'Signal', 'MACD_state', 'MACD_state_5d']].tail(50).iloc[::-1]
                df_macd_display.index = df_macd_display.index.strftime('%Y-%m-%d')
                st.dataframe(df_macd_display, use_container_width=True)

            # Basis Momentum 탭
            if df_basis is not None:
                with tab4:
                    display_recent_signal_status(df_basis, 'signal', 'Basis Momentum')
                    st.plotly_chart(plot_basis_momentum_individual(df_basis), use_container_width=True)

                    # 데이터 테이블 (내림차순)
                    st.subheader("📋 최근 데이터")
                    df_basis_display = df_basis[['Spot', 'FWD1M', 'FWD2M', 'basis', 'forward_basis', 'basis_momentum', 'signal']].tail(24).iloc[::-1]
                    df_basis_display.index = df_basis_display.index.strftime('%Y-%m-%d')
                    st.dataframe(df_basis_display, use_container_width=True)
            else:
                with tab3:
                    st.error("❌ Basis Momentum 분석을 위해서는 FWD1M, FWD2M 컬럼이 필요합니다.")

    else:
        st.info("👈 사이드바에서 Excel 파일을 업로드하여 분석을 시작하세요.")

        # 샘플 데이터 구조 안내
        st.markdown("""
        ### 📋 필요한 데이터 구조
        업로드할 Excel 파일은 다음과 같은 구조여야 합니다:

        | Date | Spot | FWD1M | FWD2M |
        |------|------|-------|-------|
        | 2015-01-02 | 1100.45 | 1102.34 | 1105.67 |
        | 2015-01-05 | 1098.23 | 1100.12 | 1103.45 |
        | ... | ... | ... | ... |

        - **Date**: 날짜 (인덱스)
        - **Spot**: 현물 환율
        - **FWD1M**: 1개월 선물 환율 (Basis Momentum 분석용)
        - **FWD2M**: 3개월 선물 환율 (Basis Momentum 분석용)
        """)

if __name__ == "__main__":
    main()
