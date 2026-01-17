#!/bin/bash
# deploy_to_binder.sh

set -e

echo "🚀 Deploying CF Visualization to Binder"

# Configuration
REPO_NAME="cf-visualization-binder"
GITHUB_USERNAME="unicogr"

# Create project directory
mkdir -p $REPO_NAME
cd $REPO_NAME

# Initialize git
git init
git branch -M main

# Create requirements.txt
cat > requirements.txt << 'EOF'
numpy>=1.21.0
scipy>=1.7.0
matplotlib>=3.5.0
pandas>=1.3.0
ipywidgets>=8.0.0
ipyvolume>=0.6.0
voila>=0.4.0
nibabel>=3.2.0
nilearn>=0.9.0
neuropythy>=0.12.0
pyyaml>=6.0
EOF

# Create postBuild
cat > postBuild << 'EOF'
#!/bin/bash
jupyter nbextension enable --py widgetsnbextension --sys-prefix
jupyter serverextension enable voila --sys-prefix
EOF
chmod +x postBuild

# Create README
cat > README.md << EOF
# Connective Field Visualization

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/$GITHUB_USERNAME/$REPO_NAME/main?urlpath=voila%2Frender%2FsurfCFviz.ipynb)

Interactive visualization of whole-brain connective field mapping.

## Usage

Click the Binder badge above to launch the interactive app.
EOF

echo "📝 Files created. Next steps:"
echo "1. Copy your modified surfCFviz.ipynb to this directory"
echo "2. (Optional) Add sample data to ./data/"
echo "3. Run: git add ."
echo "4. Run: git commit -m 'Initial commit'"
echo "5. Create GitHub repo: https://github.com/new"
echo "6. Run: git remote add origin https://github.com/$GITHUB_USERNAME/$REPO_NAME.git"
echo "7. Run: git push -u origin main"
echo "8. Go to https://mybinder.org/ and launch!"
