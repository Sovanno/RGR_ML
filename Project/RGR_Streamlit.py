import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import joblib
#import tensorflow as tf
#import keras
#import bias



def mean_absolute_percentage_error(y_true, y_pred):
    mape = np.mean(np.abs((y_true - y_pred) / y_true))
    return mape

def mean_squared_error(y_true, y_pred):
    mse = np.mean((y_true - y_pred)**2)
    return mse

def predict_for_dataset(clf, X_test, y_test):
    y_pred = clf.predict(X_test)

    st.write(f'MAPE: {mean_absolute_percentage_error(y_test, y_pred)}')
    st.write(f'MSE: {mean_squared_error(y_test, y_pred)}')

def show_author_page():
    st.title("Разработка Web-приложения (дашборда) для инференса (вывода) моделей ML и анализа данных")

    st.header("Автор")
    st.write("ФИО: Финк Эдуард Валентинович")
    st.write("Группа: ФИТ-222")
    image = "Project/This is for RGR.jpg"
    st.image(image, caption='Моё фото', use_column_width=True)

def show_dataset_page():
    st.title("Информация о наборе данных")

    st.header("Тематика набора данных")
    st.write("Этот классический набор данных содержит цены и другие атрибуты почти 54 000 бриллиантов. Это отличный набор данных для начинающих, которые учатся работать с анализом данных и визуализацией.")

    st.header("Описание признаков")
    st.write("price: цена в долларах США (\$326--\$18,823)")
    st.write("carat: вес бриллианта (0.2--5.01)")
    st.write("cut: качество среза (Fair, Good, Very Good, Premium, Ideal)")
    st.write("color: цвет бриллианта, от J (worst) до D (best)")
    st.write("clarity: измерение того, насколько чист алмаз (I1 (worst), SI2, SI1, VS2, VS1, VVS2, VVS1, IF (best))")
    st.write("x: длина в mm (0--10.74)")
    st.write("y: ширина в mm (0--58.9)")
    st.write("z: глубина в mm (0--31.8)")
    st.write("depth: общая глубина в процентах = z / mean(x, y) = 2 * z / (x + y) (43--79)")
    st.write("table: ширина вершины ромба относительно самой широкой точки (43--95)")

    st.header("Особенности предобработки данных")
    st.write("В данном наборе данных нужно было предугадывать цену бриллианта в долларах США. Сама цена находится в столбике price.")
    st.write("В наборе данных присутствовал столбик 'Unnamed: 0', который не является нужным, он был удалён.")
    st.write("Также в датасете присутствовали дубликаты, которые в последствии были удалены.")
    st.write("Столбики cut, color, clarity были представленны в виде чисел.")
    st.subheader("Для cut замена выглядит так:")
    ratings = {
        'Ideal': 5,
        'Premium': 4,
        'Very Good': 3,
        'Good': 2,
        'Fair': 1
    }
    for rating, value in ratings.items():
        st.write(f"{rating}: {value}")
    st.subheader("Для color замена выглядит так:")
    ratings = {
        'D': 7,
        'E': 6,
        'F': 5,
        'G': 4,
        'H': 3,
        'I': 2,
        'J': 1
    }
    for rating, value in ratings.items():
        st.write(f"{rating}: {value}")
    st.subheader("Для clarity замена выглядит так:")
    ratings = {
        'IF': 8,
        'VVS1': 7,
        'VVS2': 6,
        'VS1': 5,
        'VS2': 4,
        'SI1': 3,
        'SI2': 2,
        'I1': 1
    }
    for rating, value in ratings.items():
        st.write(f"{rating}: {value}")
    st.write("Был проведен EDA и удалены выбросы.")

def show_visualizations_page():
    st.title("Визуализации данных")

    data = pd.read_csv("Project/diamonds_redux.csv")
    st.subheader("Примеры визуализаций:")

    st.write("Столбчатая диаграмма:")
    plt.figure(figsize=(8, 6))
    sns.countplot(x="color", data=data)
    st.pyplot(plt)

    st.write("Ящик с усами:")
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="color", y="price", data=data)
    st.pyplot(plt)

    st.write("Тепловая карта:")
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm")
    st.pyplot(plt)

    st.write("Точечный график:")
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x="carat", y="price", data=data)
    st.pyplot(plt)

def show_predict_dataset_page():
    model1 = joblib.load("Project/LinearRegression_model.pkl")
    model2 = joblib.load("Project/Linear-one_model.pkl")
    model3 = joblib.load("Project/bagging-one_model.pkl")
    model4 = joblib.load("Project/gradient_boosting_model.pkl")
    model5 = joblib.load("Project/stacking_model.pkl")
    model6 = joblib.load("Project/NeuralNetwork.pkl")
    uploaded_file = st.file_uploader("Загрузите файл данных (CSV)", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        df.drop(df.filter(regex="Unname"),axis=1, inplace=True)
        st.write("Полученный набор данных:", df)

        models_mapping = {
            'LinearRegression(множественная)': model1,
            'LinearRegression(простая)': model2,
            'bagging': model3,
            'boosting': model4,
            'stacking': model5,
            'NeuralNetwork': model6
        }
        cut_model = st.selectbox("Выберите тип модели", list(models_mapping.keys()))
        model = models_mapping[cut_model]
        y = df["price"]
        X = df
        X.drop(['price'], axis=1, inplace=True)
        if model is model2:
            X = df["carat"]
            X = X.values.reshape(-1, 1)
        elif model is model6:
            sample = df.sample(n=1000)
            X = np.array(sample)
            y = np.array(y)
        predict_for_dataset(model, X, y)

def show_prediction_page():
    st.title("Предсказание цены алмаза")

    model1 = joblib.load("Project/LinearRegression_model.pkl")
    model2 = joblib.load("Project/Linear-one_model.pkl")
    model3 = joblib.load("Project/bagging-one_model.pkl")
    model4 = joblib.load("Project/gradient_boosting_model.pkl")
    model5 = joblib.load("Project/stacking_model.pkl")
    model6 = joblib.load("Project/NeuralNetwork.pkl")

    uploaded_file = st.file_uploader("Загрузите файл данных (CSV)", type="csv")
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        y_df = df["price"]

        st.subheader("Параметры алмаза")
        carat = st.slider("Carat", 0.2, 5.01, 1.0)
        cut_mapping = {
            'Fair': 1,
            'Good': 2,
            'Very Good': 3,
            'Premium': 4,
            'Ideal': 5
        }
        cut_text = st.selectbox("Cut Quality", list(cut_mapping.keys()))
        cut = cut_mapping[cut_text]
        color_mapping = {
            'D': 7,
            'E': 6,
            'F': 5,
            'G': 4,
            'H': 3,
            'I': 2,
            'J': 1
        }
        color_text = st.selectbox("Color", list(color_mapping.keys()))
        color = color_mapping[color_text]
        clarity_mapping = {
            'IF': 8,
            'VVS1': 7,
            'VVS2': 6,
            'VS1': 5,
            'VS2': 4,
            'SI1': 3,
            'SI2': 2,
            'I1': 1
        }
        clarity_text = st.selectbox("Clarity", list(clarity_mapping.keys()))
        clarity = clarity_mapping[clarity_text]
        x = st.slider("Length (mm)", 0.0, 10.74, 5.0)
        y = st.slider("Width (mm)", 0.0, 58.9, 3.0)
        z = st.slider("Depth (mm)", 0.0, 31.8, 2.0)
        depth = st.slider("Depth Percentage (%)", 43, 79, 60)
        table = st.slider("Table Width (%)", 43, 95, 60)

        features = np.array([[carat, cut, color, clarity, x, y, z, depth, table]])
        featuresl = np.array(carat).reshape(1, -1)
        price_prediction1 = model1.predict(features)
        price_prediction2 = model2.predict(featuresl)
        price_prediction3 = model3.predict(features)
        price_prediction4 = model4.predict(features)
        price_prediction5 = model5.predict(features)
        price_prediction6 = model6.predict(features)

        st.subheader("Предсказанная цена")
        st.write("Линейная регрессия")
        st.write("$", np.round(price_prediction1[0], 2))
        st.write("Точность MAPE", mean_absolute_percentage_error(y_df, np.round(price_prediction1[0], 2)))
        st.write("Точность MSE", mean_squared_error(y_df, np.round(price_prediction1[0], 2)))
        st.write("Простая линейная регрессия")
        st.write("$", np.round(price_prediction2[0], 2))
        st.write("Точность MAPE", mean_absolute_percentage_error(y_df, np.round(price_prediction2[0], 2)))
        st.write("Точность MSE", mean_squared_error(y_df, np.round(price_prediction2[0], 2)))
        st.write("Бэггинг")
        st.write("$", np.round(price_prediction3[0], 2))
        st.write("Точность MAPE", mean_absolute_percentage_error(y_df, np.round(price_prediction3[0], 2)))
        st.write("Точность MSE", mean_squared_error(y_df, np.round(price_prediction3[0], 2)))
        st.write("Градиентный Бустинг")
        st.write("$", np.round(price_prediction4[0], 2))
        st.write("Точность MAPE", mean_absolute_percentage_error(y_df, np.round(price_prediction4[0], 2)))
        st.write("Точность MSE", mean_squared_error(y_df, np.round(price_prediction4[0], 2)))
        st.write("Стэкинг")
        st.write("$", np.round(price_prediction5[0], 2))
        st.write("Точность MAPE", mean_absolute_percentage_error(y_df, np.round(price_prediction5[0], 2)))
        st.write("Точность MSE", mean_squared_error(y_df, np.round(price_prediction5[0], 2)))
        st.write("Нейронная сеть")
        st.write("$", np.round(price_prediction6[0], 2))
        st.write("Точность MAPE", mean_absolute_percentage_error(y_df, np.round(price_prediction6[0], 2)))
        st.write("Точность MSE", mean_squared_error(y_df, np.round(price_prediction6[0], 2)))

pages = {
    "Об авторе": show_author_page,
    "Набор данных": show_dataset_page,
    "Визуализация": show_visualizations_page,
    "Предсказаиня для набора данных":show_predict_dataset_page,
    "Предсказания": show_prediction_page
}

page = st.sidebar.selectbox("Выберите страницу", tuple(pages.keys()))

pages[page]()