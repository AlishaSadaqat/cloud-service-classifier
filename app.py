import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ── Page Config ─────────────────────────────────────────────
st.set_page_config(
    page_title="☁️ Cloud Classifier",
    page_icon="🌸",
    layout="centered"
)

# ── Custom CSS — Girly but Professional ─────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@300;400;500&display=swap');

:root {
    --rose:     #E91E8C;
    --blush:    #F8BBD9;
    --soft:     #FFF0F7;
    --mauve:    #C2185B;
    --lavender: #F3E5F5;
    --text:     #3A1C2E;
    --muted:    #9E6B85;
    --white:    #FFFFFF;
    --card:     #FFFAFD;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: var(--text);
    background-color: var(--soft);
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #FFF0F7 0%, #F3E5F5 50%, #FCE4EC 100%);
    min-height: 100vh;
}

/* Hide streamlit default elements */
#MainMenu, footer, header { visibility: hidden; }

/* Title */
h1 {
    font-family: 'Playfair Display', serif !important;
    color: var(--rose) !important;
    font-size: 2.6rem !important;
    text-align: center;
    letter-spacing: -0.5px;
    margin-bottom: 0 !important;
}

h2, h3 {
    font-family: 'Playfair Display', serif !important;
    color: var(--mauve) !important;
}

/* Subtitle */
.subtitle {
    text-align: center;
    color: var(--muted);
    font-size: 1rem;
    font-weight: 300;
    margin-bottom: 2rem;
    letter-spacing: 0.5px;
}

/* Cards */
.card {
    background: var(--card);
    border: 1px solid #F8C8DC;
    border-radius: 20px;
    padding: 1.8rem 2rem;
    margin: 1.2rem 0;
    box-shadow: 0 4px 24px rgba(233,30,140,0.07);
}

/* Divider */
hr {
    border: none;
    border-top: 1.5px solid #F8BBD9;
    margin: 1.5rem 0;
}

/* Radio buttons */
div[role="radiogroup"] label {
    background: var(--white);
    border: 1.5px solid #F8BBD9;
    border-radius: 12px;
    padding: 0.5rem 1rem;
    margin: 0.3rem 0;
    transition: all 0.2s;
    color: var(--text) !important;
    font-size: 0.9rem;
}
div[role="radiogroup"] label:hover {
    border-color: var(--rose);
    background: #FFF0F7;
}

/* Text area */
textarea {
    border: 1.5px solid #F8BBD9 !important;
    border-radius: 14px !important;
    background: var(--white) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
    color: var(--text) !important;
    padding: 1rem !important;
}
textarea:focus {
    border-color: var(--rose) !important;
    box-shadow: 0 0 0 3px rgba(233,30,140,0.1) !important;
}

/* Button */
.stButton > button {
    background: linear-gradient(135deg, #E91E8C, #C2185B) !important;
    color: white !important;
    border: none !important;
    border-radius: 50px !important;
    padding: 0.7rem 2.5rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.5px !important;
    cursor: pointer !important;
    transition: all 0.3s !important;
    box-shadow: 0 4px 15px rgba(233,30,140,0.3) !important;
    width: 100% !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(233,30,140,0.4) !important;
}

/* Dataframe */
.stDataFrame {
    border: 1px solid #F8BBD9 !important;
    border-radius: 14px !important;
    overflow: hidden;
}

/* Expander */
details {
    border: 1.5px solid #F8BBD9 !important;
    border-radius: 14px !important;
    background: var(--white) !important;
}

/* Result boxes */
.result-saas {
    background: linear-gradient(135deg, #FCE4EC, #F8BBD9);
    border-left: 5px solid #E91E8C;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin-top: 1rem;
}
.result-paas {
    background: linear-gradient(135deg, #F3E5F5, #E1BEE7);
    border-left: 5px solid #9C27B0;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin-top: 1rem;
}
.result-iaas {
    background: linear-gradient(135deg, #EDE7F6, #D1C4E9);
    border-left: 5px solid #673AB7;
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin-top: 1rem;
}
.result-title {
    font-family: 'Playfair Display', serif;
    font-size: 1.8rem;
    font-weight: 700;
    margin: 0;
}
.result-sub {
    font-size: 0.9rem;
    margin-top: 0.3rem;
    opacity: 0.75;
}

/* Badge */
.badge {
    display: inline-block;
    background: linear-gradient(135deg, #E91E8C, #C2185B);
    color: white;
    border-radius: 50px;
    padding: 0.2rem 0.9rem;
    font-size: 0.78rem;
    font-weight: 500;
    letter-spacing: 0.5px;
    margin-left: 0.5rem;
}

/* Header bar */
.top-bar {
    background: linear-gradient(135deg, #E91E8C, #C2185B);
    border-radius: 20px;
    padding: 2rem;
    text-align: center;
    margin-bottom: 1.5rem;
    box-shadow: 0 8px 32px rgba(233,30,140,0.25);
}
.top-bar h1 {
    color: white !important;
    margin: 0 !important;
    font-size: 2.2rem !important;
}
.top-bar p {
    color: rgba(255,255,255,0.85);
    margin: 0.5rem 0 0 0;
    font-size: 0.95rem;
}

/* Section labels */
.section-label {
    font-size: 0.75rem;
    font-weight: 500;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--rose);
    margin-bottom: 0.5rem;
}

/* Accuracy pill */
.acc-pill {
    background: #FCE4EC;
    color: #C2185B;
    border-radius: 50px;
    padding: 0.15rem 0.7rem;
    font-size: 0.8rem;
    font-weight: 500;
    border: 1px solid #F8BBD9;
}

/* Warning */
.stAlert {
    border-radius: 12px !important;
    border: 1.5px solid #F8BBD9 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Dataset ──────────────────────────────────────────────────
@st.cache_data
def load_data():
    data = [
        ('Gmail','Web-based email service that allows users to send receive and manage emails with cloud storage and spam filtering','SaaS'),
        ('Google Docs','Online word processor that allows real-time collaboration and document editing stored in the cloud','SaaS'),
        ('Microsoft Office 365','Suite of productivity applications including Word Excel PowerPoint hosted and delivered via the cloud','SaaS'),
        ('Salesforce CRM','Customer relationship management platform for managing sales marketing and customer support workflows','SaaS'),
        ('Dropbox','Cloud-based file storage and synchronization service for sharing files across devices and teams','SaaS'),
        ('Slack','Team communication platform with messaging channels file sharing and integrations delivered as a service','SaaS'),
        ('Zoom','Video conferencing and online meeting platform hosted in the cloud with recording and collaboration features','SaaS'),
        ('HubSpot','Marketing sales and CRM software delivered over the internet for managing customer interactions','SaaS'),
        ('Trello','Project management tool using boards and cards for task tracking delivered as a web application','SaaS'),
        ('QuickBooks Online','Cloud-based accounting software for small businesses managing invoices expenses and payroll','SaaS'),
        ('Shopify','E-commerce platform allowing businesses to create and manage online stores without infrastructure management','SaaS'),
        ('Zendesk','Customer support and helpdesk software platform delivered over the web for ticket management','SaaS'),
        ('Adobe Creative Cloud','Suite of creative applications including Photoshop Illustrator delivered via subscription model','SaaS'),
        ('Netflix','Streaming service delivering movies and TV shows on demand over the internet','SaaS'),
        ('Spotify','Music streaming platform providing access to millions of songs delivered via cloud infrastructure','SaaS'),
        ('GitHub','Web-based platform for version control and collaboration using Git repositories hosted in the cloud','SaaS'),
        ('Notion','All-in-one workspace for notes databases tasks and wikis delivered as a cloud application','SaaS'),
        ('Asana','Work management platform for teams to track projects tasks and workflows via the cloud','SaaS'),
        ('Monday.com','Work operating system for managing projects workflows and team collaboration as a cloud service','SaaS'),
        ('Figma','Browser-based collaborative interface design tool for creating UI UX designs in real time','SaaS'),
        ('Canva','Online graphic design platform for creating presentations social media graphics and marketing materials','SaaS'),
        ('LinkedIn Learning','Online learning platform with video courses on professional skills delivered via the cloud','SaaS'),
        ('DocuSign','Electronic signature and agreement cloud platform for signing and managing documents digitally','SaaS'),
        ('ServiceNow','IT service management platform automating enterprise workflows delivered as a cloud application','SaaS'),
        ('Workday','Human capital management and financial management software delivered as a cloud service','SaaS'),
        ('Google App Engine','Platform for building and deploying scalable web applications with automatic infrastructure management','PaaS'),
        ('Microsoft Azure App Service','Fully managed platform for building hosting and scaling web apps and APIs in the cloud','PaaS'),
        ('Heroku','Cloud platform that lets developers build run and operate applications without managing infrastructure','PaaS'),
        ('AWS Elastic Beanstalk','Service for deploying and scaling web applications automatically handling capacity provisioning and load balancing','PaaS'),
        ('Red Hat OpenShift','Kubernetes-based container platform for developing and deploying cloud-native applications','PaaS'),
        ('Google Firebase','Mobile and web application development platform with real-time database authentication and hosting','PaaS'),
        ('Salesforce Platform','Development platform for building and deploying custom business applications in the Salesforce ecosystem','PaaS'),
        ('IBM Cloud Foundry','Open source platform for deploying and scaling applications without managing underlying servers','PaaS'),
        ('Oracle Cloud Platform','Integrated platform services for developing testing and deploying cloud applications with built-in AI','PaaS'),
        ('Pivotal Cloud Foundry','Enterprise platform for continuous delivery of applications with automated scaling and management','PaaS'),
        ('Render','Cloud platform for deploying web services databases and static sites with automatic builds and scaling','PaaS'),
        ('Railway','Infrastructure platform for deploying applications with built-in databases and automatic deployments','PaaS'),
        ('Fly.io','Platform for running applications close to users worldwide using edge deployment infrastructure','PaaS'),
        ('Netlify','Development platform for building deploying and managing modern web projects with CI/CD pipelines','PaaS'),
        ('Vercel','Frontend cloud platform for deploying web applications with serverless functions and global CDN','PaaS'),
        ('AWS Lambda','Serverless computing platform that runs code in response to events without provisioning servers','PaaS'),
        ('Google Cloud Functions','Event-driven serverless compute platform for running code without managing infrastructure','PaaS'),
        ('Azure Functions','Serverless compute service for executing event-driven code without provisioning or managing servers','PaaS'),
        ('Cloudflare Workers','Serverless execution environment that runs JavaScript at the edge without managing servers','PaaS'),
        ('Supabase','Open source Firebase alternative providing database authentication and storage as a platform service','PaaS'),
        ('MongoDB Atlas','Fully managed cloud database service for deploying MongoDB with built-in automation and scaling','PaaS'),
        ('PlanetScale','Serverless MySQL-compatible database platform with branching and automatic scaling capabilities','PaaS'),
        ('CockroachDB Cloud','Distributed SQL database platform offering resilience and scalability without operational overhead','PaaS'),
        ('Snowflake','Cloud data platform for storing and analyzing large datasets with elastic scalability and sharing','PaaS'),
        ('Databricks','Unified analytics platform based on Apache Spark for big data processing and machine learning','PaaS'),
        ('Amazon EC2','Scalable virtual servers in the cloud allowing users to configure and manage compute capacity on demand','IaaS'),
        ('Azure Virtual Machines','On-demand scalable computing resources in Azure allowing deployment of Windows and Linux VMs','IaaS'),
        ('Google Compute Engine','Virtual machine instances running in Googles data centers with customizable CPU memory and storage','IaaS'),
        ('IBM Cloud Virtual Servers','Scalable virtual server instances with flexible compute profiles for running workloads in IBM Cloud','IaaS'),
        ('Oracle Cloud Infrastructure','Bare metal and virtual machine instances with high performance networking and storage for enterprise workloads','IaaS'),
        ('DigitalOcean Droplets','Simple scalable virtual machines for developers with flexible compute options and SSD storage','IaaS'),
        ('Vultr Cloud Compute','High-performance cloud compute instances with global data center locations and hourly billing','IaaS'),
        ('Linode by Akamai','Linux virtual machine instances with dedicated CPU and memory for developer and production workloads','IaaS'),
        ('Amazon S3','Scalable object storage service for storing and retrieving any amount of data from anywhere','IaaS'),
        ('Google Cloud Storage','Unified object storage for storing and accessing data in Google infrastructure with global availability','IaaS'),
        ('Azure Blob Storage','Massively scalable object storage for unstructured data including text binary images and backups','IaaS'),
        ('Amazon EBS','Block storage volumes for use with EC2 instances providing persistent low-latency storage','IaaS'),
        ('Google Persistent Disk','Reliable high performance block storage for virtual machine instances in Google Cloud','IaaS'),
        ('Amazon VPC','Isolated virtual network in AWS for launching cloud resources in a defined virtual networking environment','IaaS'),
        ('Azure Virtual Network','Private network in Azure for securely connecting resources with routing firewalls and subnets','IaaS'),
        ('Google Cloud VPC','Global virtual private cloud network for managing networking across Google Cloud resources','IaaS'),
        ('AWS CloudFormation','Infrastructure as code service for provisioning and managing AWS resources using templates','IaaS'),
        ('Terraform Cloud','Infrastructure automation platform for provisioning and managing cloud infrastructure using code','IaaS'),
        ('AWS Direct Connect','Dedicated network connection from on-premises data center to AWS for consistent network performance','IaaS'),
        ('Azure ExpressRoute','Private dedicated connectivity between on-premises networks and Microsoft Azure data centers','IaaS'),
        ('Amazon RDS','Managed relational database service supporting MySQL PostgreSQL Oracle and SQL Server in the cloud','IaaS'),
        ('Amazon EC2 Auto Scaling','Automatically adjusts number of EC2 instances based on demand to maintain performance and reduce cost','IaaS'),
        ('Google Kubernetes Engine','Managed Kubernetes service for running containerized applications in Google Cloud infrastructure','IaaS'),
        ('Azure Kubernetes Service','Managed Kubernetes container orchestration service for deploying and managing containerized workloads','IaaS'),
        ('Amazon EKS','Managed Kubernetes service for running Kubernetes clusters in AWS without managing control plane','IaaS'),
        ('Rackspace Cloud Servers','Managed cloud hosting with virtual server infrastructure and support for enterprise workloads','IaaS'),
        ('Alibaba Cloud ECS','Elastic compute service offering virtual machines with flexible CPU memory and storage configurations','IaaS'),
        ('Tencent Cloud CVM','Cloud virtual machine service providing secure and reliable computing capacity in Tencent infrastructure','IaaS'),
        ('OVHcloud Public Cloud','European cloud infrastructure provider offering virtual machines storage and networking services','IaaS'),
        ('Hetzner Cloud','Cost-effective cloud infrastructure with virtual servers and dedicated servers for developers','IaaS'),
        ('Scaleway Instances','French cloud provider offering virtual compute instances with ARM and x86 architectures','IaaS'),
        ('Fastly CDN','Content delivery network providing edge cloud infrastructure for fast and secure content delivery','IaaS'),
        ('Cloudflare CDN','Global content delivery network providing DDoS protection performance and security infrastructure','IaaS'),
        ('Amazon CloudFront','Fast content delivery network integrating with AWS services for delivering data to users globally','IaaS'),
        ('AWS IAM','Identity and access management service for securely controlling access to AWS services and resources','IaaS'),
        ('Azure Active Directory','Cloud-based identity and access management service for managing users and controlling resource access','IaaS'),
        ('Google Cloud IAM','Identity and access management for controlling who has access to Google Cloud resources','IaaS'),
        ('AWS Load Balancer','Automatically distributes incoming application traffic across multiple targets for fault tolerance','IaaS'),
        ('Azure Load Balancer','High availability service distributing network traffic across healthy virtual machines in Azure','IaaS'),
        ('Google Cloud Load Balancing','Fully distributed managed load balancing service scaling to handle user traffic spikes globally','IaaS'),
        ('Amazon Glacier','Low cost cloud storage service for long-term data archival and backup with infrequent access','IaaS'),
        ('Azure Archive Storage','Lowest cost storage tier for rarely accessed data with high durability and flexible retrieval options','IaaS'),
        ('Google Cloud Filestore','Fully managed NFS file storage for applications requiring a filesystem interface and shared storage','IaaS'),
        ('AWS Route 53','Scalable domain name system web service for routing users to internet applications reliably','IaaS'),
        ('Cloudflare DNS','Fast and secure DNS service providing domain name resolution with DDoS protection','IaaS'),
        ('Azure DNS','Hosting service for DNS domains providing name resolution using Microsoft Azure infrastructure','IaaS'),
    ]
    return pd.DataFrame(data, columns=['service_name','description','category'])

@st.cache_resource
def train_models():
    df = load_data()
    le = LabelEncoder()
    y  = le.fit_transform(df['category'])
    tfidf = TfidfVectorizer(max_features=500, stop_words='english', ngram_range=(1,2), sublinear_tf=True)
    X_tfidf = tfidf.fit_transform(df['description']).toarray()
    chi2_sel = SelectKBest(chi2, k=100)
    X_chi2   = chi2_sel.fit_transform(X_tfidf, y)
    pca = PCA(n_components=30, random_state=42)
    X_pca = pca.fit_transform(X_chi2)
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42, stratify=y)
    models = {
        'KNN (K=5)':           KNeighborsClassifier(n_neighbors=5),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'SVM (RBF Kernel)':    SVC(kernel='rbf', C=1.0, random_state=42),
        'Decision Tree':       DecisionTreeClassifier(max_depth=10, random_state=42),
    }
    trained = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        acc = accuracy_score(y_test, m.predict(X_test))
        trained[name] = {'model': m, 'accuracy': acc}
    return tfidf, chi2_sel, pca, le, trained

# ── Header ───────────────────────────────────────────────────
st.markdown("""
<div class="top-bar">
    <h1>🌸 Cloud Service Classifier</h1>
    <p>Classify cloud services into SaaS · PaaS · IaaS using Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# ── Load Models ──────────────────────────────────────────────
with st.spinner("✨ Loading models..."):
    tfidf, chi2_sel, pca, le, trained_models = train_models()

model_names = list(trained_models.keys())

# ── Model Selector ───────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<p class="section-label">🤖 Select Model</p>', unsafe_allow_html=True)

labels = []
for n in model_names:
    acc = trained_models[n]['accuracy']*100
    labels.append(f"{n}  —  {acc:.0f}% accuracy")

selected_label = st.radio("", labels, index=2, label_visibility="collapsed")
selected_model = model_names[labels.index(selected_label)]
st.markdown('</div>', unsafe_allow_html=True)

# ── Input ────────────────────────────────────────────────────
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<p class="section-label">📝 Enter Service Description</p>', unsafe_allow_html=True)
user_input = st.text_area(
    "",
    placeholder="e.g. Virtual machine instances with configurable CPU and RAM for running workloads in the cloud...",
    height=130,
    label_visibility="collapsed"
)

classify_btn = st.button("🔍 Classify Service")
st.markdown('</div>', unsafe_allow_html=True)

# ── Result ───────────────────────────────────────────────────
if classify_btn:
    if not user_input.strip():
        st.warning("💭 Please enter a service description first!")
    else:
        x = tfidf.transform([user_input]).toarray()
        x = chi2_sel.transform(x)
        x = pca.transform(x)
        pred = le.inverse_transform(trained_models[selected_model]['model'].predict(x))[0]

        config = {
            'SaaS': ('result-saas', '💼', '#E91E8C', 'Software as a Service — Ready-to-use application delivered over the internet'),
            'PaaS': ('result-paas', '🛠️', '#9C27B0', 'Platform as a Service — Development & deployment platform without server management'),
            'IaaS': ('result-iaas', '🖥️', '#673AB7', 'Infrastructure as a Service — Virtualized compute, storage & networking resources'),
        }
        cls, emoji, color, desc = config[pred]
        st.markdown(f"""
        <div class="{cls}">
            <p class="result-title" style="color:{color}">{emoji} {pred}</p>
            <p class="result-sub" style="color:{color}">{desc}</p>
            <p class="result-sub" style="margin-top:0.8rem">Model used: <strong>{selected_model}</strong></p>
        </div>
        """, unsafe_allow_html=True)

st.markdown('<hr>', unsafe_allow_html=True)

# ── Accuracy Table ───────────────────────────────────────────
st.markdown('<p class="section-label">📊 Model Accuracies</p>', unsafe_allow_html=True)
acc_df = pd.DataFrame({
    'Model': model_names,
    'Test Accuracy': [f"{trained_models[m]['accuracy']*100:.1f}%" for m in model_names],
    'Status': ['⭐ Best' if trained_models[m]['accuracy'] >= 0.90 and m == 'SVM (RBF Kernel)' else '' for m in model_names]
})
st.dataframe(acc_df, use_container_width=True, hide_index=True)

st.markdown('<hr>', unsafe_allow_html=True)

# ── Dataset Preview ──────────────────────────────────────────
with st.expander("🌸 View Dataset (first 10 rows)"):
    st.dataframe(load_data().head(10), use_container_width=True, hide_index=True)

st.markdown("""
<p style="text-align:center; color:#C2185B; font-size:0.8rem; margin-top:1.5rem; opacity:0.7">
    🌸 Cloud Computing Project · Alisha Sadaqat · BCS223095 · CUST Islamabad
</p>
""", unsafe_allow_html=True)
