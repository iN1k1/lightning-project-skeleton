#!/bin/bash

RESET="\033[0m"
BOLD="\033[1m"
CYAN="\033[36m"
GREEN="\033[92m"

# Prompt the user for the GitHub repository name and the library repository name
read -p "$(echo $BOLD$CYAN"Enter the new GitHub url: "$RESET)" github_repo_url
read -p "$(echo $BOLD$CYAN"Enter the library name (now is lightning_project_skeleton): "$RESET)" library_repo_name

# Get the current directory
current_dir=$(pwd)

# Move the main folder to a new folder with the library repository name
cd ..
cp -r "$current_dir" "$library_repo_name"
cd "$library_repo_name"

# Replace every occurrence of "lightning_project_skeleton" with the GitHub repository name in all files
grep -rl --include=*.py "lightning_project_skeleton" . | xargs sed -i '' "s/lightning_project_skeleton/$library_repo_name/g"

# Rename the folder named "lightning_project_skeleton" within the "src" subfolder
if [ -d "src/lightning_project_skeleton" ]; then
    mv src/lightning_project_skeleton "src/$library_repo_name"
fi

# Change the URL in setup.py
sed -i '' 's|author="Niki Martinel"|author="New Author"|g' setup.py
sed -i '' 's|author_email="niki.martinel@gmail.com"|author_email="newauthor@xyz.abc"|g' setup.py
sed -i '' 's|url="https://github.com/iN1k1/lightning-project-skeleton"|url="$github_repo_url"|g' setup.py

# Refresh the README.md
echo "# $library_repo_name" > README.md
echo "This project has been renamed to $library_repo_name." >> README.md
echo "The new GitHub repository URL is $github_repo_url." >> README.md
echo "Derived from https://github.com/iN1k1/lightning-project-skeleton" >> README.md


## GitHub updates
git remote rm origin
git add ./*
git commit -m "Initial commit after renaming"
git branch -M master
git remote add origin $github_repo_url
git push -u origin master

# All done!
echo $BOLD$GREEN"Project has been renamed and moved successfully!"$RESET