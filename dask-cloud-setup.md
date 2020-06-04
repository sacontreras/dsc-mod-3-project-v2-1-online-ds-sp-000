## Create Kubertes cluser in GCP

### Step 1: Change to appropriate zone
<code>
gcloud config set compute/zone us-west1
</code>

### Step 2: Create container
This assumes you already have the appropriate quotas provisioned in order to use the <code>n1-highcpu-8</code> machine.

<code>
gcloud container clusters create dask-cluster-1 \
    --num-nodes 3 \
    --machine-type n1-highcpu-8 \
    --scopes bigquery,storage-rw \
    --preemptible \
    --no-async
</code>

### Step 3: Connect to container in GCP terminal
<code>
gcloud container clusters get-credentials dask-cluster-1 --region us-west1 --project data-science-278317
</code>

### Step 4: Set up Helm
#### Step 4.1: Get Heml
<code>
curl https://raw.githubusercontent.com/kubernetes/helm/master/scripts/get | bash
</code>

#### Step 4.2: Setup Heml Service Account
<code>
kubectl --namespace kube-system create serviceaccount tiller
</code>

#### Step 4.3: Give the ServiceAccount full permissions to manage the cluster
<code>
kubectl create clusterrolebinding tiller --clusterrole cluster-admin --serviceaccount=kube-system:tiller
</code>

#### Step 4.4: Initialize helm and tiller
<code>
helm init --service-account tiller --history-max 100 --wait
</code>

<p/><p/>
<code>
helm init --client-only
</code>

#### Step 4.5: Verify
<code>
helm version
</code>

<p/><p/>
<code>
helm init --client-only
</code>

### Step 5: Set up JupyterHub
#### Step 5.1: Generate a random hex string representing 32 bytes to use as a security token
<code>
openssl rand -hex 32
</code>

#### Step 5.2: Create and start editing a file called config.yaml
<code>
nano config.yaml
</code>

#### Step 5.3: Write the following into the config.yaml file but instead of writing &lt;RANDOM-HEX&gt; paste the generated hex string you copied in step 5.1
<pre>
proxy:
  secretToken: "&lt;RANDOM_HEX&gt;"
</pre>

REMOVE THIS LATER:
proxy:
  secretToken: "5744c6c3d0cf6a52ed98055adc5601f7c0073551b33a16395eb42757b8a7db76"

#### Step 5.4: Install JupyterHub
##### Step 5.4.1: Make Helm aware of the JupyterHub Helm chart repository so you can install the JupyterHub chart from it without having to use a long URL name
<code>
helm repo add jupyterhub https://jupyterhub.github.io/helm-chart/
</code>

<p/><p/>
<code>
helm repo update
</code>

##### Step 5.4.2: Now install the chart configured by your config.yaml by running this command from the directory that contains your config.yaml
<pre>
# Suggested values: advanced users of Kubernetes and Helm should feel
# free to use different values.
RELEASE=jhub
NAMESPACE=jhub

helm upgrade --install $RELEASE jupyterhub/jupyterhub \
  --namespace $NAMESPACE  \
  --version=0.8.2 \
  --values config.yaml
</pre>

##### Step 5.4.3: You can see the pods created by entering
<code>
kubectl get pod --namespace jhub
</code>

##### Step 5.4.4: Wait for the hub and proxy pod to enter the <i>Running</i> state

##### Step 5.4.5: Find the IP we can use to access the JupyterHub. Run the following command until the EXTERNAL-IP of the proxy-public service is available like in the example output
<code>
kubectl get service --namespace jhub
</code>

##### Step 5.4.6: To use JupyterHub, enter the external IP for the proxy-public service in to a browser. JupyterHub is running with a default dummy authenticator so entering any username and password combination will let you enter the hub

34.83.187.220

80:31816/TCP,443:30551/TCP

### Step 6: Helm Install Dask
#### Step 6.1: Dask maintains a Helm chart repository containing various charts for the Dask community https://helm.dask.org/ . You will need to add this to your known channels and update your local charts
<code>
helm repo add dask https://helm.dask.org/
</code>

<p/><p/>
<code>
helm repo update
</code>

#### Step 6.2: Now, you can launch Dask on your Kubernetes cluster using the Dask Helm chart
<code>
helm install dask/dask
</code>

#### Step 6.3: Verify Deployment
<code>
kubectl get pods
</code>

<p/><p/>
<code>
kubectl get services
</code>


Jupyter:    34.105.52.151 (80:30714/TCP)
Scheduler:  35.230.13.87 (8786:31509/TCP,80:31794/TCP)
