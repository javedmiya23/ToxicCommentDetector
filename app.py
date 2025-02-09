# File: app.py

import os
import pandas as pd
import tensorflow as tf
import numpy as np
import streamlit as st
from urllib.parse import urlparse, parse_qs
import plotly.express as px
from collections import Counter
from wordcloud import STOPWORDS
import requests
import re

# Load the pre-trained model for toxicity detection
model = tf.keras.models.load_model('toxicity.h5')

# Load the dataset and initialize the vectorizer
df = pd.read_csv(os.path.join('analysis', 'train.csv', 'train.csv'))
X = df['comment_text']
MAX_FEATURES = 200000
vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=MAX_FEATURES,
    output_sequence_length=1800,
    output_mode='int'
)
vectorizer.adapt(X.values)

# Function to score a comment's toxicity and related parameters
def score_comment(comment):
    """
    Analyze the comment and return a dictionary of toxicity parameters.
    Returns scores for attributes such as toxic, threat, insult, etc.
    """
    vectorized_comment = vectorizer([comment])
    results = model.predict(vectorized_comment)[0]  # Output is an array of scores
    attributes = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    return {attr: score for attr, score in zip(attributes, results)}

# Extract IDs from the given Facebook URL
def extract_ids_from_url(url):
    """
    Extracts Page ID/Name and Post ID from the given Facebook URL.
    Supports formats:
    - https://www.facebook.com/{page_id}/posts/{post_id}
    - https://www.facebook.com/{page_name}/posts/{post_id}
    - https://www.facebook.com/photo?fbid={post_id}&set=a.{other_id}
    """
    if "photo" in url and "fbid" in url:
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)
        post_id = query_params.get("fbid", [None])[0]  # Extract 'fbid' parameter
        return None, post_id  # Page identifier not present in this format
    else:
        match = re.match(r"https://www\.facebook\.com/([^/]+)/posts/([^/?]+)", url)
        if match:
            page_identifier = match.group(1)  # Page ID or Name
            post_id = match.group(2)
            return page_identifier, post_id
    return None, None

# Fetch comments using the Graph API
def get_comments(page_identifier, post_id, access_token):
    """
    Fetch comments for a given Facebook post using the Graph API.
    Includes the author information (user who posted the comment).
    """
    object_id = f"{page_identifier}_{post_id}" if page_identifier else post_id
    api_url = f"https://graph.facebook.com/v12.0/{object_id}/comments"
    params = {
        "access_token": access_token,
        "fields": "message,from{name,id}"  # Request user info along with the comment
    }

    response = requests.get(api_url, params=params)

    if response.status_code == 200:
        data = response.json()
        comments = [{"message": comment["message"], "user": comment.get("from", {}).get("name", "Unknown")} 
                    for comment in data.get("data", [])]
        return comments
    else:
        return None, response.json()

# Save comments to a CSV file
def save_comments_to_csv(post_id, comments):
    """
    Save comments to a CSV file with the post ID.
    """
    file_path = os.path.join('analysis', 'comments', 'data.csv')
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    file_exists = os.path.isfile(file_path)
    data = [{"post_id": post_id, "comment": comment["message"], "user": comment["user"]} 
            for comment in comments]
    df = pd.DataFrame(data)
    df.to_csv(file_path, mode="a", index=False, header=not file_exists)

# Load dataset
DATASET_PATH = r"D:\DATA SCIENCE\CommentToxicity\analysis\train.csv\train.csv"

@st.cache_data
def load_data(file_path):
    """Load data from the CSV file."""
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        st.error(f"File not found at {file_path}. Please check the path and try again.")
        return None

def sanitize_word(word):
    """Sanitize inappropriate words by replacing inner characters with '*'."""
    if len(word) > 2:
        return word[0] + '*' * (len(word) - 2) + word[-1]
    return word  # Leave short words unchanged

# Modify get_top_words function to sanitize words
def get_top_words(comments, top_n=3):
    """Extract top N words from comments."""
    stop_words = set(STOPWORDS)
    all_words = " ".join(comments).split()
    filtered_words = [word.lower() for word in all_words if word.lower() not in stop_words]
    word_counts = Counter(filtered_words)
    sanitized_counts = [(sanitize_word(word), count) for word, count in word_counts.most_common(top_n)]
    return sanitized_counts

# Path to save the flagged data
output_file = os.path.join('analysis', 'flagged', 'data.csv')

# Function to save data to CSV
def save_to_csv(data):
    # Check if the file exists
    file_exists = os.path.isfile(output_file)
    # Convert data to a DataFrame
    df = pd.DataFrame([data])
    # Append to the file if it exists, otherwise create it
    df.to_csv(output_file, mode='a', header=not file_exists, index=False)

# Streamlit App Menu
st.sidebar.title("Menu")
menu_option = st.sidebar.selectbox("Choose an Option:", [
    "Home",
    "Data Analysis",
    "Comment Analyzer",
    "Comment Retriever with Author",
    "Comment Retriever with Toxicity and Author"
])

if menu_option == "Home":
    st.title("Welcome to the Vulnerable and Toxic Comment Analysis Tool")
    st.markdown("""
    ### About the Tool
    This application helps users analyze Facebook post comments for toxicity and provides visual insights.
    
    **Key Features**:
    - **Data Analysis**: Visualize toxicity data with advanced plots.
    - **Comment Analyzer**: Assess individual comment toxicity levels.
    - **Comment Retriever with Author**: Fetch comments along with their authors.
    - **Comment Retriever with Toxicity and Author**: Fetch and analyze comments for toxicity and display toxicity levels.

    ### Instructions
    - Use the **sidebar menu** to navigate between features.
    - Enter the required inputs like a comment or Facebook post URL in the provided fields.
    - Follow on-screen instructions to analyze, visualize, or save the data.

    **Note**: Ensure you have a valid Facebook API access token for comment retrieval.
    """)

elif menu_option == "Data Analysis":
        # Streamlit app
    st.title("Comment Toxicity Analysis: Combined Visualizations")

    # Load and display dataset
    df = load_data(DATASET_PATH)
    if df is not None:
        st.sidebar.success("Dataset Loaded!")

        # Process data for first 2 plots
        comment_counts = {
            "Category": ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"],
            "No. of Comments": [
                df["toxic"].sum(),
                df["severe_toxic"].sum(),
                df["obscene"].sum(),
                df["threat"].sum(),
                df["insult"].sum(),
                df["identity_hate"].sum(),
            ],
        }
        comment_counts_df = pd.DataFrame(comment_counts)

        total_comments = len(df)
        non_toxic = total_comments - (df[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].sum(axis=1) > 0).sum()
        partially_toxic = (df[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].sum(axis=1) == 1).sum()
        fully_toxic = total_comments - non_toxic - partially_toxic

        toxicity_summary = {
            "Toxicity Level": ["Non-Toxic", "Partially Toxic", "Fully Toxic"],
            "No. of Comments": [non_toxic, partially_toxic, fully_toxic],
        }
        toxicity_summary_df = pd.DataFrame(toxicity_summary)

        # Plot 1: Number of Comments by Toxicity Categories
        st.subheader("Plot 1: Number of Comments by Toxicity Categories")
        fig_3d_1 = px.scatter_3d(
            comment_counts_df,
            x="Category",
            y="No. of Comments",
            z=[0] * len(comment_counts_df),  # All points have z=0 for 2D appearance
            color="Category",
            title="3D Plot: Number of Comments by Category",
        )
        st.plotly_chart(fig_3d_1)

        fig_2d_1 = px.bar(
            comment_counts_df,
            x="Category",
            y="No. of Comments",
            color="Category",
            title="2D Bar Graph: Number of Comments by Category",
        )
        st.plotly_chart(fig_2d_1)

        # Plot 2: Analysis by Toxicity Levels
        st.subheader("Plot 2: Analysis by Toxicity Levels")
        fig_3d_2 = px.scatter_3d(
            toxicity_summary_df,
            x="Toxicity Level",
            y="No. of Comments",
            z=[0] * len(toxicity_summary_df),
            color="Toxicity Level",
            title="3D Plot: Toxicity Levels",
        )
        st.plotly_chart(fig_3d_2)

        fig_2d_2 = px.bar(
            toxicity_summary_df,
            x="Toxicity Level",
            y="No. of Comments",
            color="Toxicity Level",
            title="2D Bar Graph: Toxicity Levels",
        )
        st.plotly_chart(fig_2d_2)

        # Plot 3: Common Toxic Words vs Total Comments
        toxic_comments = df[df["toxic"] > 0]["comment_text"].dropna()
        top_toxic_words = get_top_words(toxic_comments, top_n=10)
        top_toxic_words_df = pd.DataFrame(top_toxic_words, columns=["Word", "Count"])

        st.subheader("Plot 3: Common Toxic Words vs Total Comments")
        fig_3d_3 = px.scatter_3d(
            top_toxic_words_df,
            x="Word",
            y="Count",
            z=[0] * len(top_toxic_words_df),
            color="Word",
            title="3D Plot: Common Toxic Words",
        )
        st.plotly_chart(fig_3d_3)

        fig_2d_3 = px.bar(
            top_toxic_words_df,
            x="Word",
            y="Count",
            color="Count",
            title="2D Bar Graph: Common Toxic Words",
        )
        st.plotly_chart(fig_2d_3)

        # Plot 4: Top 3 Words for Each Toxicity Category
        st.subheader("Plot 4: Top 3 Words for Each Toxicity Category")
        toxicity_categories = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
        top_words_data = []

        for category in toxicity_categories:
            category_comments = df[df[category] > 0]["comment_text"].dropna()
            top_words = get_top_words(category_comments, top_n=3)
            for word, count in top_words:
                top_words_data.append({"Category": category, "Word": word, "Count": count})

        top_words_df = pd.DataFrame(top_words_data)

        fig_3d_4 = px.scatter_3d(
            top_words_df,
            x="Category",
            y="Count",
            z=[0] * len(top_words_df),
            color="Word",
            title="3D Plot: Top 3 Words for Each Toxicity Category",
        )
        st.plotly_chart(fig_3d_4)

        fig_2d_4 = px.bar(
            top_words_df,
            x="Category",
            y="Count",
            color="Word",
            title="2D Bar Graph: Top 3 Words for Each Toxicity Category",
            barmode="group",
        )
        st.plotly_chart(fig_2d_4)
    else:
        st.warning("Unable to load data. Please verify the file path.")


elif menu_option == "Comment Retriever with Author":

    st.title("Social Media Comment Retriever")
    
    # Common inputs for both tools
    post_url = st.text_input("Enter the Facebook Post URL:")
    access_token = st.text_input("Enter your Page Access Token:", type="password")

    # Initialize session state
    if "comments" not in st.session_state:
        st.session_state.comments = None
    if "post_id" not in st.session_state:
        st.session_state.post_id = None

    st.header("Comment Retriever")
    if st.button("Fetch Comments"):
        if post_url and access_token:
            page_identifier, post_id = extract_ids_from_url(post_url)

            if not post_id:
                st.error("Invalid URL format. Please provide a valid Facebook post URL.")
            else:
                st.info("Fetching comments... This may take a moment.")
                result = get_comments(page_identifier, post_id, access_token)

                if isinstance(result, list):
                    if result:
                        st.success(f"Retrieved {len(result)} comments:")
                        for i, comment in enumerate(result, start=1):
                            st.write(f"{i}. {comment['message']} - Posted by: {comment['user']}")
                        st.session_state.comments = result
                        st.session_state.post_id = post_id
                    else:
                        st.warning("No comments found for this post.")
                else:
                    st.error("Failed to fetch comments.")
                    st.json(result)
        else:
            st.error("Please provide both the Facebook Post URL and Page Access Token.")
    if st.session_state.comments and st.session_state.post_id:
        if st.button("Save Comments"):
            save_comments_to_csv(st.session_state.post_id, st.session_state.comments)
            st.success(f"Comments saved!")

elif menu_option == "Comment Analyzer":
    # Streamlit UI
    st.title("Toxic Comment Detector")
    st.write("Enter a comment below to analyze its toxicity levels:")

    # Input Textbox
    comment = st.text_area("Comment to Score", placeholder="Type your comment here...")

    # Initialize session state for results
    if "result" not in st.session_state:
        st.session_state.result = None

    # Analyze Button
    if st.button("Analyze Comment"):
        if comment.strip():
            with st.spinner("Analyzing..."):
                toxicity_scores = score_comment(comment)  # Correctly capture return value
                st.session_state.result = toxicity_scores
            st.success("Analysis Complete!")
            st.write("### Toxicity Scores")
            for attr, score in st.session_state.result.items():
                st.write(f"- **{attr.capitalize()}**: {score:.2f}")
        else:
            st.warning("Please enter a valid comment!")

    # Save Data Button
    if st.session_state.result:
        if st.button("Save Data"):
            save_to_csv({"comment": comment, **st.session_state.result})
            st.success("Data saved to analysis/flagged/data.csv!")


elif menu_option == "Comment Retriever with Toxicity and Author":

    st.title("Social Media Comment Retriever And Analyzer")
    post_url = st.text_input("Enter the Facebook Post URL:")
    access_token = st.text_input("Enter your Page Access Token:", type="password")

    # Initialize session state
    if "comments" not in st.session_state:
        st.session_state.comments = None
    if "post_id" not in st.session_state:
        st.session_state.post_id = None

    st.header("Comment Retriever and Analyzer")
    if st.button("Fetch Comments and Analyze"):
        if post_url and access_token:
            page_identifier, post_id = extract_ids_from_url(post_url)

            if not post_id:
                st.error("Invalid URL format. Please provide a valid Facebook post URL.")
            else:
                st.info("Fetching comments... This may take a moment.")
                result = get_comments(page_identifier, post_id, access_token)

                if isinstance(result, list):
                    if result:
                        st.success(f"Retrieved {len(result)} comments:")
                        st.session_state.comments = result
                        st.session_state.post_id = post_id
                        for i, comment in enumerate(result, start=1):
                            toxicity_scores = score_comment(comment["message"])
                            st.write(f"{i}. {comment['message']} (Posted by: {comment['user']})")
                            st.write(f"   Toxicity Scores:")
                            for attr, score in toxicity_scores.items():
                                st.write(f"      - {attr}: {score:.2f}")
                            if toxicity_scores['toxic'] > 0.5:
                                st.write("   This comment is likely toxic!")
                            else:
                                st.write("   This comment is not toxic.")
                    else:
                        st.warning("No comments found for this post.")
                else:
                    st.error("Failed to fetch comments.")
                    st.json(result)
        else:
            st.error("Please provide both the Facebook Post URL and Page Access Token.")
    if st.session_state.comments and st.session_state.post_id:
        if st.button("Save Comments"):
            save_comments_to_csv(st.session_state.post_id, st.session_state.comments)
            st.success(f"Comments saved!")
        
    
