import os
import re
import joblib
import openai
import streamlit as st
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# ---------- 설정 (필요 시 경로 수정) ----------
DATA_PATH = "./AI를 활용한 성적 예측 모델 제작용 설문조사 (Responses).xlsx"
MODEL_PATH = "./coach_model.joblib"
RANDOM_STATE = 42
TARGET_COL = "match" # 데이터에 이미 존재하면 사용, 없으면 생성

# ---------- 유틸: 모의고사 앞 3합 계산 ----------
def sum_first3_digits(val):
    if pd.isna(val):
        return np.nan
    s = str(val)
    digits = re.findall(r"\d", s)
    if not digits:
        return np.nan
    first3 = digits[:3]
    return sum(int(d) for d in first3)

# ---------- 데이터 로드 및 전처리 ----------
@st.cache_data(ttl=600)
def load_and_prepare(data_path=DATA_PATH):
    if not os.path.exists(data_path):
        return None, f"데이터 파일이 없습니다: {data_path}"

    try:
        df = pd.read_excel(data_path)
    except Exception as e:
        return None, f"엑셀 파일을 읽는 중 오류: {e}"

    # 안전하게 컬럼 존재 확인 및 전처리
    # 1) 내신 numeric
    if '귀하의 내신 성적이 무엇입니까?' in df.columns:
        df['내신_numeric'] = pd.to_numeric(df['귀하의 내신 성적이 무엇입니까?'], errors='coerce')
    else:
        df['내신_numeric'] = np.nan

    # 2) 모의고사 앞 3합
    if '귀하의 모의고사 성적이 무엇입니까?' in df.columns:
        df['모의_first3_sum'] = df['귀하의 모의고사 성적이 무엇입니까?'].apply(sum_first3_digits)
    else:
        df['모의_first3_sum'] = np.nan

    # 3) 상위권 라벨 생성(있으면 사용)
    if TARGET_COL in df.columns:
        try:
            df[TARGET_COL] = df[TARGET_COL].astype(int)
        except:
            df[TARGET_COL] = ((df['내신_numeric'] <= 1.6) | (df['모의_first3_sum'] <= 5)).astype(int)
    else:
        df[TARGET_COL] = ((df['내신_numeric'] <= 1.6) | (df['모의_first3_sum'] <= 5)).astype(int)

    # 주요 후보 컬럼 (존재하면 사용)
    candidate_cols = [
        '내신_numeric', '모의_first3_sum',
        '평일에 공부를 얼마나 하십니까?(순공시간, 학원 제외)',
        '주말에 공부를 얼마나 하십니까?(순공시간, 학원 제외)',
        '집중 시간', '전자기기 사용 시간', '아침식사를 하십니까?', '루틴이 있습니까?',
        '계획을 세웁니까?(플래너 등)', '오답노트를 작성 하십니까?', '공부할 때 음악을 듣습니까?',
        '학원을 총 몇 개 다니시나요?', '학원 숙제를 얼마나 합니까?', '결석/지각을 얼마나 하십니까?',
        '방과후, 심화탐구 등의 활동을 하시나요?', '학교 수업에서 얼마나 집중하시나요?', '자습할 때 얼마나 집중하시나요?',
        '카페인을 섭취합니까?'
    ]
    cols = [c for c in candidate_cols if c in df.columns]
    df_model = df[cols + [TARGET_COL]].copy()

    return (df_model, df)

# ---------- 모델 학습/로드 ----------
@st.cache_resource
def get_or_train_model(df_model):
    # df_model: DataFrame with features + TARGET_COL
    if os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
            return model, "loaded"
        except Exception:
            pass

    X = df_model.drop(columns=[TARGET_COL])
    y = df_model[TARGET_COL]

    # numeric / categorical 구분
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    num_transform = SimpleImputer(strategy="median")
    cat_transform = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", num_transform, numeric_cols),
        ("cat", cat_transform, categorical_cols)
    ], remainder="drop")

    clf = Pipeline(steps=[
        ("pre", preprocessor),
        ("clf", RandomForestClassifier(n_estimators=200, random_state=RANDOM_STATE))
    ])

    clf.fit(X, y)
    joblib.dump(clf, MODEL_PATH)
    return clf, "trained"

# ---------- 규칙 기반 간단 조언 생성 ----------
def generate_advice(model, user_input_df):
    out_lines = []
    try:
        prob = model.predict_proba(user_input_df)[0][1]
        pred = model.predict(user_input_df)[0]
        out_lines.append(f"상위권 예측 확률: {prob*100:.1f}% (모델 판정: {'상위권' if pred==1 else '상위권 아님'})")
    except Exception:
        out_lines.append("모델 예측을 수행할 수 없습니다. 입력형식 또는 필드 확인 필요.")

    # 규칙형 팁 (간단)
    try:
        weekday = user_input_df.iloc[0].get('평일에 공부를 얼마나 하십니까?(순공시간, 학원 제외)')
        weekend = user_input_df.iloc[0].get('주말에 공부를 얼마나 하십니까?(순공시간, 학원 제외)')
        focus = user_input_df.iloc[0].get('집중 시간')
    except Exception:
        weekday = weekend = focus = None

    if isinstance(weekday, str):
        if any(x in weekday for x in ['1', '2']) and not any(x in weekday for x in ['3','4','5','6','7']):
            out_lines.append("평일 공부시간이 적을 수 있습니다. 하루 1~2시간이라도 루틴화하세요.")
        else:
            out_lines.append("평일 공부시간은 양호합니다. 꾸준함 유지가 중요합니다.")
    if isinstance(weekend, str) and any(x in weekend for x in ['5','6','7','8','9','10']):
        out_lines.append("주말 공부시간이 충분합니다. 휴식도 잊지 마세요.")
    if isinstance(focus, str) and any(x in focus for x in ['1','2']):
        out_lines.append("집중 지속 시간이 1~2시간 수준이면 포모도로(25/5)도 고려하세요.")

    out_lines.append("(모델 참고) 주요 영향: 학습시간·루틴·집중도 등")
    return "\n".join(out_lines)

# ---------- OpenAI GPT 연동 함수 ----------
def gpt_advice(api_key, profile_text, user_question=None, model_name="gpt-3.5-turbo"):
    if not api_key:
        return "OpenAI API 키가 제공되지 않았습니다."
    openai.api_key = api_key

    system = ("당신은 친절하고 실용적인 한국어 AI 학습 코치입니다. "
              "학생 프로필과 모델 요약을 보고, 구체적이고 실행 가능한 학습 조언을 3~6개 항목으로 정리해 주세요. "
              "포모도로, 루틴, 오답노트, 과목별 우선순위 등 실전 팁을 포함하세요.")
    user_msg = f"학생 프로필:\n{profile_text}"
    if user_question:
        user_msg += f"\n학생 질문: {user_question}"

    try:
        resp = openai.ChatCompletion.create(
            model=model_name,
            messages=[
                {"role":"system", "content": system},
                {"role":"user", "content": user_msg}
            ],
            max_tokens=600,
            temperature=0.7
        )
        return resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"GPT 호출 오류: {e}"

# ---------- Streamlit UI ----------
st.set_page_config(page_title="AI 학습코치 (GPT 통합)", layout="wide")
st.title("AI 학습코치 — GPT 통합 데모")

# 데이터 로드
with st.spinner("데이터 로드 중..."):
    loaded = load_and_prepare()
    if loaded is None:
        st.error("데이터 로드 실패")
        st.stop()
    df_model, df_or_msg = loaded if isinstance(loaded, tuple) else (None, "로드 오류")
    if df_model is None:
        st.error(df_or_msg)
        st.stop()

# 모델 로드/학습
with st.spinner("모델 확인/학습 중..."):
    model_obj, status = get_or_train_model(df_model)
st.sidebar.write(f"모델 상태: {status}")
st.sidebar.write(f"모델 파일: {MODEL_PATH}")

# 사이드바: OpenAI API 키 입력 (선택)
st.sidebar.header("설정")
api_key_input = st.sidebar.text_input("OpenAI API 키 (선택) — 입력하면 GPT 상담이 활성화됩니다", type="password")
api_key_env = api_key_input if api_key_input else os.environ.get("OPENAI_API_KEY", "")

# 사용자 입력 폼 (동적으로 모델에 쓰이는 컬럼 기준)
st.sidebar.header("학생 정보 입력 (테스트)")
input_cols = [c for c in df_model.columns if c != TARGET_COL]
user_vals = {}
for c in input_cols:
    # 빈칸 허용. 사용자가 직접 입력
    user_vals[c] = st.sidebar.text_input(c, "")

# 예측 버튼
if st.sidebar.button("예측 및 조언 생성"):
    # 1행 DataFrame 생성
    X_input = pd.DataFrame([{c: (np.nan if user_vals[c]=="" else user_vals[c]) for c in input_cols}])
    # 모델에 맞춰 숫자형 컬럼 변환(가능하면)
    for col in X_input.columns:
        if pd.api.types.is_numeric_dtype(df_model[col]):
            # 시도 변환
            try:
                X_input[col] = pd.to_numeric(X_input[col], errors="coerce")
            except:
                pass
    advice_text = generate_advice(model_obj, X_input)
    st.subheader("예측 및 규칙 기반 조언")
    st.code(advice_text)

# GPT 상담 섹션
st.subheader("GPT 상담 (옵션)")
st.write("학생 정보 입력 후, 모델 요약과 함께 GPT에게 자연어 상담을 요청할 수 있습니다.")
user_question = st.text_input("GPT에게 물어볼 질문 예: '집중이 안돼요. 어떻게 할까요?'")
if st.button("GPT 상담 요청"):
    # 준비: 프로필 텍스트 생성
    X_input = pd.DataFrame([{c: (np.nan if user_vals[c]=="" else user_vals[c]) for c in input_cols}])
    # 모델 기반 요약
    model_summary = generate_advice(model_obj, X_input)
    profile_lines = []
    for c in input_cols:
        profile_lines.append(f"{c}: {X_input.iloc[0].get(c)}")
    profile_text = "\n".join(profile_lines) + "\n\n모델 요약:\n" + model_summary

    if not api_key_env:
        st.error("OpenAI API 키가 필요합니다. 사이드바에 키를 입력하거나 환경변수 OPENAI_API_KEY를 설정하세요.")
    else:
        with st.spinner("GPT에게 질의 중..."):
            gpt_resp = gpt_advice(api_key_env, profile_text, user_question)
        st.subheader("GPT 상담 결과")
        st.write(gpt_resp)

# 원본 데이터 미리보기
if st.checkbox("원 데이터 샘플 보기 (상단 50개)"):
    st.dataframe(df_model.head(50))

st.markdown("---")
st.write("주의: 이 앱은 PoC(개념증명) 수준입니다. 실제 서비스 배포 전에는 입력 검증, 개인정보보호, 모델 검증(교차검증 등)이 필요합니다.")


