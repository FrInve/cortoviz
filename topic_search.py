import streamlit as st
import pandas as pd
from bertopic import BERTopic
import seaborn as sns
import matplotlib.pyplot as plt
from src.visualization.timeline import *
from src.visualization.wc import create_wordcloud
from datetime import datetime
from scipy.stats import kruskal

st.set_page_config(layout="wide")

@st.cache_data
def load_cases_data():
    df_covid_cases = get_covid_data(end_date='2022-06-15')
    return df_covid_cases

@st.cache_data
def load_timeline_data():
    covid_timeline = pd.read_csv('./data/raw/macmillan_covid.csv')
    covid_timeline['Date'] = pd.to_datetime(covid_timeline['Date'])
    covid_timeline.set_index("Date",inplace=True)
    return covid_timeline

@st.cache_data
def load_topic_data():
    df_norm = pd.read_csv('./data/processed/topics_freq_pivoted.csv')
    df_norm['Date'] = pd.to_datetime(df_norm['Date'])
    df_norm = df_norm.set_index('Date')
    return df_norm

@st.cache_data
def load_topic_abs_data():
    df_abs_freq = pd.read_parquet('./data/processed/topics.parquet')
    df_abs_freq['Date'] = df_abs_freq.Timestamp.dt.strftime('%Y/%m/%d')
    df_abs_freq['Date'] = pd.to_datetime(df_abs_freq['Date'])
    df_abs_freq['Topic'] = df_abs_freq.Topic.astype(str)
    df_abs_freq = df_abs_freq.loc[df_abs_freq.index.repeat(df_abs_freq.Frequency)]
    df_abs_freq = df_abs_freq[['Date','Topic']]
    df_abs_freq = df_abs_freq.set_index('Date')
    return df_abs_freq

@st.cache_resource
def load_topic_model():
    topic_model = BERTopic.load('./models/BERTopic_full_2023-04-18',embedding_model="pritamdeka/BioBERT-mnli-snli-scinli-scitail-mednli-stsb")
    return topic_model

@st.cache_resource
def create_wordcloud_static(topic):
    return create_wordcloud(topic_model, topic).to_image()


df_covid_cases = load_cases_data()
covid_timeline = load_timeline_data()
df_norm = load_topic_data()
df_abs_freq = load_topic_abs_data()
topic_model = load_topic_model()

st.title("CORToViz")
st.header("The CORD-19 Topics Visualizer")

wcol1, wcol2 = st.columns(2)

with wcol1:
    query = st.text_input("Search a topic:", value='variant', max_chars=100)

    similar_topic, similarity = topic_model.find_topics(query, top_n=6)
    #st.text("Topic\tSimilarity")
    #for topic, sim in zip(similar_topic, similarity):
    #    st.text(f"{topic}\t{sim}")

    st.subheader("Inspect Topics' WordClouds:")

    col1, col2, col3 = st.columns(3)
    columns = [col1,col2,col3]

    selecting_topics = {str(topic):True for topic in similar_topic}


    for i, topic in enumerate(similar_topic):
        with columns[i%3]:
            st.text(f"{topic_model.get_topic(topic)[0][0]}")
            st.image(create_wordcloud_static(topic))
            chosen_val = st.checkbox(str(topic),value=True)
            selecting_topics[str(topic)] = chosen_val
            st.text("------------------")
        

    old_selecting_topics = selecting_topics.copy()
    with col3:
        stat_selected_topic = st.selectbox(
            "Select one topic to verify if it changes through time (Kruskal-Wallis Test):",
            similar_topic
        )
        if st.checkbox("Show only this topic", value=False):
            selecting_topics = {str(topic):(True if str(topic)==str(stat_selected_topic) else False) for topic in similar_topic}
        else:
            selecting_topics = old_selecting_topics
            
        
    selected_topics = [topic[0] for topic in selecting_topics.items() if topic[1]]

with wcol2:
    st.text(f"Currently showing {len(selected_topics)} topics over time")

    df_tmp = df_norm[df_norm.index < '2022-06-06'][selected_topics].copy()
    df_abs_tmp = df_abs_freq[(df_abs_freq.Topic.isin(selected_topics)) & (df_abs_freq.index < '2022-06-06')].copy()
    # Set same colour palette among different plots
    palette_colors = sns.color_palette('tab10')
    palette_dict = {topic:color for topic,color in zip(selected_topics,palette_colors)}
    fig, (ax1, ax1_bis) = plt.subplots(2,height_ratios=[0.87,0.13],figsize=(11.7,8.27))
    #sns.set_theme()
    #plt.stackplot(df_tmp.index, df_tmp.iloc[:,0] )
    sns.lineplot(data=df_tmp, dashes=False, palette=palette_dict, ax=ax1).set(title=query, ylabel="Relative Frequency")
    sns.histplot(data=df_abs_tmp, multiple="stack", x="Date",hue="Topic", palette=palette_dict, legend=False).set(ylabel="Absolute Frequency")
    ax2= ax1.twinx()
    plot_events(covid_timeline[covid_timeline.Included==1][['Event']].to_dict()['Event'])
    plot_covid_cases(covid_df=df_covid_cases, ax=ax2)
    st.pyplot(fig)

    stat_first_date_ranges = st.slider(
        "Select the first time interval (YY/MM/DD)",
        value=(datetime(2020,3,1,0,0), datetime(2020,9,1,0,0)),
        format="YY/MM/DD",
        min_value=datetime(2020,1,1,0,0),
        max_value=datetime(2022,6,6,0,0)
    )
    stat_first_mask = (df_tmp.index >= stat_first_date_ranges[0]) & (df_tmp.index <= stat_first_date_ranges[1])
    stat_first_samples = df_tmp[[str(stat_selected_topic)]][stat_first_mask].to_numpy(na_value=0)
    st.write(f"Number of samples: {len(stat_first_samples)}")

    stat_second_date_ranges = st.slider(
        "Select the second time interval (YY/MM/DD)",
        value=(datetime(2021,6,1,0,0), datetime(2021,12,1,0,0)),
        format="YY/MM/DD",
        min_value=datetime(2020,1,1,0,0),
        max_value=datetime(2022,6,6,0,0)
    )
    stat_second_mask = (df_tmp.index >= stat_second_date_ranges[0]) & (df_tmp.index <= stat_second_date_ranges[1])
    stat_second_samples = df_tmp[[str(stat_selected_topic)]][stat_second_mask].to_numpy(na_value=0)
    st.write(f"Number of samples: {len(stat_second_samples)}")

    r_col1, r_col2 = st.columns(2)
    stat_kruskal = kruskal(stat_first_samples, stat_second_samples)
    with r_col1:
        if stat_kruskal.pvalue > 0.05:
            st.write("There is no statistically significant difference (p-value=5%)")
        else:
            st.write("There is statistically significant difference (p-value=5%)")
    with r_col2:
        st.write(f"p-value: {stat_kruskal.pvalue[0]:.5f}")
        st.write(f"H statistic: {stat_kruskal.statistic[0]:.5f}")

st.markdown("Copyright (C) 2023 Francesco Invernici, All Rights Reserved")