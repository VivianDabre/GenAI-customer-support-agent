import streamlit as st
import requests

st.title("Customer Support Assistant")

with st.form("query_form"):
    name = st.text_input("Name")
    age = st.number_input("Age", min_value=1, max_value=120)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    email = st.text_input("Email")
    subject = st.text_input("Subject")
    query = st.text_area("Your Query")

    submitted = st.form_submit_button("Submit")

if submitted:
    with st.spinner("Processing..."):
        payload = {
            "name": name,
            "age": age,
            "gender": gender,
            "email": email,
            "subject": subject,
            "query": query
        }
        try:
            response = requests.post("http://localhost:8000/fetch-data", json=payload)
            if response.status_code == 200:
                result = response.json()
                st.success("Query processed successfully!")
                st.markdown("### Steps Taken:")
                for step in result.get("steps", []):
                    st.write(step)

                st.markdown(f"### Final Resolution:\n{result.get('final_resolution')}")
            else:
                st.error("Failed to process: " + str(response.text))

        except Exception as e:
            st.error(f"Request failed: {e}")
