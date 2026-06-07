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

# ─── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Cloud Service Classifier",
    page_icon="☁️",
    layout="centered"
)

# ─── Dataset ───────────────────────────────────────────────────
@st.cache_data
def load_data():
    descriptions = [
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
        ('Hetzner Cloud','Cost-effective cloud infrastructure with virtual servers and dedicated servers for developers and businesses','IaaS'),
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
    df = pd.DataFrame(descriptions, columns=['service_name', 'description', 'category'])
    return df

@st.cache_resource
def train_models():
    df = load_data()
    le = LabelEncoder()
    y = le.fit_transform(df['category'])

    tfidf = TfidfVectorizer(max_features=500, stop_words='english', ngram_range=(1,2), sublinear_tf=True)
    X_tfidf = tfidf.fit_transform(df['description']).toarray()

    chi2_selector = SelectKBest(chi2, k=100)
    X_chi2 = chi2_selector.fit_transform(X_tfidf, y)

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

    return tfidf, chi2_selector, pca, le, trained

# ─── UI ────────────────────────────────────────────────────────
st.title("☁️ Cloud Service Classifier")
st.markdown("**Classify cloud services into SaaS, PaaS, or IaaS using Machine Learning**")
st.divider()

# Load models
with st.spinner("Training models..."):
    tfidf, chi2_selector, pca, le, trained_models = train_models()

# Model selector
st.subheader("🤖 Select Model")
model_names = list(trained_models.keys())
model_accs  = [f"{trained_models[m]['accuracy']*100:.0f}%" for m in model_names]
labels      = [f"{n}  —  Test Acc: {a}" for n, a in zip(model_names, model_accs)]

selected_label = st.radio("", labels, index=2)  # default: SVM
selected_model_name = model_names[labels.index(selected_label)]

st.divider()

# Input
st.subheader("📝 Enter Service Description")
user_input = st.text_area(
    "Describe the cloud service:",
    placeholder="e.g. Virtual machine instances with configurable CPU and RAM for running workloads in the cloud",
    height=120
)

# Predict
if st.button("🔍 Classify", use_container_width=True):
    if user_input.strip() == "":
        st.warning("Please enter a description first.")
    else:
        x = tfidf.transform([user_input]).toarray()
        x = chi2_selector.transform(x)
        x = pca.transform(x)
        model = trained_models[selected_model_name]['model']
        pred  = le.inverse_transform(model.predict(x))[0]

        color_map = {'SaaS': '#4A90D9', 'PaaS': '#7ED321', 'IaaS': '#F5A623'}
        emoji_map = {'SaaS': '💼', 'PaaS': '🛠️', 'IaaS': '🖥️'}

        st.markdown(f"""
        <div style="background:{color_map[pred]}22; border-left: 5px solid {color_map[pred]};
                    padding: 20px; border-radius: 8px; margin-top:10px;">
            <h2 style="color:{color_map[pred]}; margin:0">{emoji_map[pred]} Predicted: {pred}</h2>
            <p style="margin:8px 0 0 0; color:#555">Model used: <b>{selected_model_name}</b></p>
        </div>
        """, unsafe_allow_html=True)

st.divider()

# Model accuracy table
st.subheader("📊 Model Accuracies")
acc_df = pd.DataFrame({
    'Model': model_names,
    'Test Accuracy': [f"{trained_models[m]['accuracy']*100:.1f}%" for m in model_names]
})
st.dataframe(acc_df, use_container_width=True, hide_index=True)

st.divider()

# Dataset preview
with st.expander("📂 View Dataset (first 10 rows)"):
    st.dataframe(load_data().head(10), use_container_width=True, hide_index=True)

st.caption("Cloud Computing Course Project | Capital University of Science & Technology")
