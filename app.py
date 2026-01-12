import streamlit as st
import os
import time
from pathlib import Path

# Import your modules
# Ensure you have a file named __init__.py in the modules folder
from modules import module_1, module_2, module_3, module_4, module_5, module_6

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="ResearchAI Agent",
    page_icon="🤖",
    layout="wide"
)

# --- HEADER ---
st.title("🤖 Automated Research Assistant")
st.markdown("""
This agent discovers, reads, analyzes, and writes research papers for you.
**Enter a topic below to start the pipeline.**
""")

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuration")
    topic = st.text_input("Research Topic", "Sun")
    num_papers = st.slider("Number of Papers", 1, 10, 3)
    
    st.divider()
    
    if st.button("🚀 Start Research Mission", type="primary"):
        st.session_state['running'] = True
        st.session_state['logs'] = []

# --- MAIN LOGIC ---
if 'running' not in st.session_state:
    st.session_state['running'] = False

if st.session_state['running']:
    
    # Create tabs for the workflow
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "1. Discovery", "2. PDFs", "3. Analysis", "4. Draft", "5. Final Review"
    ])

    # --- STAGE 1: SEARCH ---
    with tab1:
        st.subheader("🔍 Discovery Phase")
        with st.status("Searching Semantic Scholar...", expanded=True) as status:
            st.write(f"Querying topic: '{topic}'...")
            try:
                # Call Module 1
                search_results, save_path = module_1.main_search(topic) # You might need to adjust arguments based on your module_1 code
                if search_results:
                    st.success(f"Found {search_results['total_results']} papers.")
                    st.json(search_results['papers'][:3])  # Show top 3
                    status.update(label="Search Complete!", state="complete", expanded=False)
                else:
                    st.error("No results found.")
                    st.stop()
            except Exception as e:
                st.error(f"Error in Module 1: {e}")
                st.stop()

    # --- STAGE 2: DOWNLOAD ---
    with tab2:
        st.subheader("📥 Acquisition Phase")
        with st.status("Downloading PDFs...", expanded=True) as status:
            try:
                # Call Module 2
                downloaded = module_2.main_download_process(count=num_papers)
                if downloaded:
                    for paper in downloaded:
                        if paper.get('downloaded'):
                            st.write(f"✅ Downloaded: {paper['title'][:50]}...")
                    status.update(label="Downloads Complete!", state="complete", expanded=False)
                else:
                    st.warning("No PDFs could be downloaded.")
            except Exception as e:
                st.error(f"Error in Module 2: {e}")

    # --- STAGE 3 & 4: EXTRACT & ANALYZE ---
    with tab3:
        st.subheader("🧠 Analysis Phase")
        col1, col2 = st.columns(2)
        
        with col1:
            st.info("Extracting Text...")
            # Call Module 3
            extracted = module_3.run_extraction_pipeline(max_papers=num_papers)
            st.write(f"Processed {len(extracted)} papers.")
            
        with col2:
            st.info("Generating Insights...")
            # Call Module 4
            analysis = module_4.main_analysis()
            if analysis:
                st.write("### Key Findings")
                # Display dynamic findings if available, else static example
                if 'comparison' in analysis.get('data', {}):
                    common = analysis['data']['comparison'].get('common_methods', [])
                    st.write(f"**Common Methods:** {', '.join(common)}")

    # --- STAGE 5: DRAFT ---
    with tab4:
        st.subheader("✍️ Drafting Phase")
        with st.spinner("Writing academic draft..."):
            # Call Module 5
            draft_results = module_5.run_draft_generation()
            
            if draft_results:
                sections = draft_results['sections']
                st.markdown("### Generated Abstract")
                st.info(sections['abstract']['content'])
                
                with st.expander("View Full Introduction"):
                    st.markdown(sections['introduction']['content'])

    # --- STAGE 6: FINAL OUTPUT ---
    with tab5:
        st.subheader("✅ Final Refined Output")
        
        # Call Module 6
        final_pack = module_6.run_revision_cycle(iterations=1)
        
        # Load the final markdown file content
        outputs_dir = Path("outputs")
        md_files = list(outputs_dir.glob("*.md"))
        
        if md_files:
            latest_md = max(md_files, key=lambda f: f.stat().st_mtime)
            md_content = latest_md.read_text(encoding='utf-8')
            
            st.markdown(md_content)
            
            st.download_button(
                label="📥 Download Final Report (Markdown)",
                data=md_content,
                file_name="Research_Report.md",
                mime="text/markdown"
            )
        else:
            st.warning("Draft generation pending...")

    st.success("Pipeline Finished Successfully!")
    st.session_state['running'] = False
