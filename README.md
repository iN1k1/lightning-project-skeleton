Lightning Skeleton Library
========================

In order to create a new library change the following files:

## Clone this repo
```git clone git@github.com:iN1k1/lightning-project-skeleton.git```

## rename the repo
```mv lightning-project-skeleton my-library```

## Git-crypt unlock
```git-crypt unlock```

## create your github repo
for example: my-library

## update the git remote of your repo
```shell
git remote rm origin
git remote add origin git@github.com:iN1k1/lightning-project-skeleton.git
```

## remove the existing terraform state
```shell
rm -rf state
```

## Setup.py
Update name, description and url

## requirements.txt
Add your dependencies

## README.md
Update the name of the library

## config/defaults.yaml
update the `component` and `source_repository_url`

## Package name
rename `src/lightning_project_skeleton` to the package name you are willing to publish.

## Add your source code
Add your source code

## Update pipeline semver config
change `key: pysatis-example-library/metadata/version` to your pipeline name in `config/pipeline/manage-defauly.yml`


# Development

## Code guidelines
- We use black formatter to format code and enforce style
- Use strong types everywhere you can
