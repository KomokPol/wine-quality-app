import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

import matplotlib.pyplot as plt
from sklearn import metrics


APP_DIR = Path(__file__).parent


@st.cache_resource
def load_model():
    return joblib.load(APP_DIR / 'model.joblib')

@st.cache_data
def load_test_data():
    return pd.read_csv(APP_DIR / 'test.csv')

@st.cache_data
def test_proba(_model, X):
    return _model.predict_proba(X)[:, 1]


model = load_model()
test_data = load_test_data()

config = json.loads(
    (APP_DIR / 'config.json').read_text(encoding='utf-8')
)

feature_names = test_data.drop(columns='target').columns.tolist()

st.title('Оценка качества вина')

threshold = st.sidebar.slider(
    'Порог рекомендации',
    min_value=0.0,
    max_value=1.0,
    value=float(config['threshold']),
    step=0.01,
)

tab_single, tab_metrics, tab_batch, tab_weights = st.tabs([
    'Оценка одного вина',
    'Модельные метрики',
    'Пакетное предсказание',
    'Что выучила модель',
])

with tab_single:
    st.write('Введите характеристики вина.')

    values = {}

    for feature in feature_names:
        if feature == 'is_red':
            color = st.radio(
                'Цвет вина',
                ['Белое', 'Красное'],
            )
            values[feature] = int(color == 'Красное')
        else:
            values[feature] = st.number_input(
                feature,
                min_value=float(test_data[feature].min()),
                max_value=float(test_data[feature].max()),
                value=float(test_data[feature].median()),
                format='%.4f',
            )

    wine = pd.DataFrame([values], columns=feature_names)

    probability = model.predict_proba(wine)[0, 1]

    st.metric(
        'Вероятность хорошего вина',
        f'{probability:.1%}',
    )

    if probability >= threshold:
        st.success('Рекомендовать вино')
    else:
        st.warning('Не рекомендовать вино')

with tab_metrics:
    X_test = test_data[feature_names]
    y_test = test_data['target']

    probabilities = test_proba(model, X_test)
    predictions = (probabilities >= threshold).astype(int)

    precision = metrics.precision_score(y_test, predictions, zero_division=0)
    recall = metrics.recall_score(y_test, predictions, zero_division=0)
    f1 = metrics.f1_score(y_test, predictions, zero_division=0)

    st.metric('Precision', f'{precision:.3f}')
    st.metric('Recall', f'{recall:.3f}')
    st.metric('F1', f'{f1:.3f}')

    cm = metrics.confusion_matrix(y_test, predictions, labels=[0, 1])

    fig, ax = plt.subplots()
    metrics.ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=['Остальные', 'Хорошие'],
    ).plot(ax=ax, cmap='Blues', values_format='d')

    ax.set_title('Матрица ошибок')
    st.pyplot(fig)
    plt.close(fig)

    tn, fp, fn, tp = cm.ravel()
    current_fpr = fp / (fp + tn)

    fpr_curve, tpr_curve, _ = metrics.roc_curve(y_test, probabilities)

    fig, ax = plt.subplots()
    ax.plot(fpr_curve, tpr_curve)
    ax.plot([0, 1], [0, 1], '--', color='gray')
    ax.scatter(current_fpr, recall, color='red', zorder=3)
    ax.set_xlabel('FPR')
    ax.set_ylabel('TPR - Recall')
    ax.set_title('ROC-кривая')
    st.pyplot(fig)
    plt.close(fig)

    precision_curve, recall_curve, _ = metrics.precision_recall_curve(y_test, probabilities)

    fig, ax = plt.subplots()
    ax.plot(recall_curve, precision_curve)
    ax.axhline(y_test.mean(), linestyle='--', color='gray')
    ax.scatter(recall, precision, color='red', zorder=3)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('PR-кривая')
    st.pyplot(fig)
    plt.close(fig)

    st.write(
        'При увеличении порога в подборку попадает меньше вин, и '
        'можно пропустить больше хороших, но обычно доля ошибочных '
        'рекомендаций уменьшается. При снижении порога подборка '
        'расширяется, но в нее попадает больше остальных вин'
    )
    st.caption('Красная точка показывает текущий порог')

with tab_batch:
    st.write('Загрузите CSV с признаками вин без столбца target')

    uploaded_file = st.file_uploader('Выберите CSV-файл', type=['csv'])

    if uploaded_file is not None:
        uploaded_data = pd.read_csv(uploaded_file)

        missing_columns = [
            column for column in feature_names
            if column not in uploaded_data.columns
        ]

        if missing_columns:
            st.error('В файле не хватает столбцов: ' + ', '.join(missing_columns))
        elif uploaded_data.empty:
            st.error('В файле нет строк с данными')
        else:
            X_uploaded = uploaded_data[feature_names]

            batch_probabilities = model.predict_proba(X_uploaded)[:, 1]

            result = uploaded_data.copy()
            result['probability'] = batch_probabilities
            result['prediction'] = (batch_probabilities >= threshold).astype(int)

            st.dataframe(result)

            st.download_button(
                label='Скачать результат',
                data=result.to_csv(index=False).encode('utf-8-sig'),
                file_name='wine_predictions.csv',
                mime='text/csv',
            )

with tab_weights:
    coefficients = pd.Series(
        model.named_steps['logisticregression'].coef_[0],
        index=feature_names,
    ).sort_values()

    fig, ax = plt.subplots()
    coefficients.plot.barh(ax=ax)
    ax.axvline(0, color='gray')
    ax.set_xlabel('Коэффициент после масштабирования')
    ax.set_title('Коэффициенты логистической регрессии')
    st.pyplot(fig)
    plt.close(fig)

    st.write('Сильнее всего в пользу хорошего вина: ' + ', '.join(coefficients.tail(3).index[::-1]))
    st.write('Сильнее всего против хорошего вина: ' + ', '.join(coefficients.head(3).index))
    st.write(
        'Положительный коэффициент повышает вероятность хорошего '
        'вина при увеличении признака, отрицательный снижает, '
        'если остальные признаки остаются неизменными'
    )
