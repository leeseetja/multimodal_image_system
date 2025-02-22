import streamlit as st
import requests
import os

def main():
    st.title("Multi-Modal Image Retrieval by Lesetja Lekoloane for SBSA")
    st.write("Enter a text query to retrieve relevant images from our dataset.")

    # If the Flask API is local:
    BACKEND_URL = "http://localhost:5000"

    query = st.text_input("Text query:")
    k = st.slider("Number of images (k):", 1, 10, 5)

    if st.button("Search"):
        payload = {"query": query, "k": k}
        try:
            response = requests.post(f"{BACKEND_URL}/search", json=payload)
            if response.status_code == 200:
                data = response.json()
                st.write("Search Results:")
                for item in data["results"]:
                    st.write(f"Path: {item['image_path']}")
                    st.write(f"Similarity: {item['similarity']:.4f}")
                    # Attempt to display the local image
                    # If the path is accessible from the local dev environment:
                    st.image(item["image_path"])
            else:
                st.error("API Error: " + response.text)
        except Exception as e:
            st.error(f"Error connecting to Flask API: {e}")

if __name__ == "__main__":
    main()

