import streamlit as st

st.title('Yoo\'s Playground')

st.divider()
st.subheader("소개")
st.markdown("""
    Yoo\'s Playground에 오신 것을 환영합니다.\n
    이곳은 제가 공부하면서 배운 내용을 실제로 적용해보고 테스트해보는 공간입니다.\n
    아직은 많이 부족하지만, 조금씩 더 나아지는 모습을 보여드리겠습니다.\n
""")
st.divider()

st.subheader("Sentifl LLM")
st.write("""
    Sentifl LLM은 졸업작품 "감정기반 BGM 생성 블로그 : Sentifl"에서 만들었던 AI로 문장을 요약하고 감정을 분류하여 그에 적합한 노래를 생성합니다.\n
    여기서는 노래생성을 제외한 요약과 감정분류 기능을 테스트해볼 수 있습니다.\n
""")