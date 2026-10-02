#!/bin/bash

cp /opt/course-runtime/Dockerfile /dli/task/composer/Dockerfile

# remove Jupyter announcements
jupyter labextension disable "@jupyterlab/apputils-extension:announcements"

mkdir /dli/task/clean_notebooks && cp /dli/task/*.ipynb /dli/task/clean_notebooks

echo "starting jupyter in the background"
jupyter lab \
        --ip 0.0.0.0                               `# Run on localhost` \
        --allow-root                               `# Enable the use of sudo commands in the notebook` \
        --LabApp.expose_app_in_browser=True \
        --no-browser                               `# Do not launch a browser by default` \
        --IdentityProvider.cookie_options='{"path":"/"}' \
        --NotebookApp.base_url="/lab"              `# Allow value to be passed in for production` \
        --NotebookApp.token="$JUPYTER_TOKEN"       `# Do not require token to access notebook` \
        --NotebookApp.password=""                  `# Do not require password to run jupyter server`
