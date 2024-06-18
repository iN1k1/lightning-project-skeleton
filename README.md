Lightning Skeleton Library
============================

In order to create a new library change the following files:

## Clone this repo
```git clone git@github.com:iN1k1/lightning-project-skeleton.git```

## rename the repo
```mv lightning-project-skeleton my-library```

## create your github repo
for example: my-library

## update the git remote of your repo
```shell
git remote rm origin
git remote add origin git@github.com:YOUR_USER/my-library.git
```

## Setup.py
Update name, description and url

## requirements.txt
Add your dependencies

## README.md
Update the name of the library

## Package name
rename `src/lightning_project_skeleton` to `src/my-library` (or any other package name you are willing to publish).

## Add your source code
Add your source code


# Development

## Code guidelines
- We generally use black formatter to format code and enforce style (you are encouraged to do the same :))
- Use strong types everywhere you can!
