# This script is designed to work on MAC (only tested on Monterey version 12.2.1 with an intel i5)
# 
sed -ie "s/torch==([0-9]+\.[0-9]+\.[0-9]+)/torch==1.4.0/g" Value-based-methods/python/requirements.txt
if { conda env list | grep 'RUN_ENV'; } >/dev/null 2>&1; then
    conda env update --file env.yml --prune
    echo "exists"
else
    conda env create -f env.yml
    echo "doesnt exist"
fi

cd Value-based-methods/python
pip install .
cd ../..

python -m ipykernel install --user --name drl_navigation --display-name "drl_navigation"

curl -O  https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip
unzip Banana.app.zip
rm Banana.app.zip