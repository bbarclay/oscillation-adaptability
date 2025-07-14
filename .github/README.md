# GitHub Configuration Files

This directory contains configuration files for GitHub features and integrations.

## Contents

- **workflows/**: GitHub Actions workflow configurations
  - `pages.yml`: Workflow for deploying GitHub Pages

- **DISCUSSION_TEMPLATE/**: Templates for GitHub Discussions
  - `general.yml`: Template for general discussions
  - `research_collaboration.yml`: Template for research collaboration proposals

- **assets/**: Visual assets for GitHub
  - `header.svg`: SVG header image for the repository
  - `social-preview.html`: HTML template for generating social preview images

- **images/**: Additional images used in GitHub documentation
  - `header.md`: Markdown version of the header for embedding
  - `header.txt`: ASCII art version of the header

- **topics.yml**: GitHub repository topics configuration

## Usage

### Social Preview Image

The `social-preview.html` file can be used to generate a social preview image for the repository. To generate the image:

1. Open the HTML file in a web browser
2. Take a screenshot of the rendered page (1280x640px)
3. Upload the screenshot as the social preview image in the repository settings

### GitHub Discussions

The discussion templates in the `DISCUSSION_TEMPLATE` directory are used to provide structured templates for GitHub Discussions. To enable GitHub Discussions:

1. Go to the repository settings
2. Scroll down to the "Features" section
3. Check the "Discussions" checkbox
4. Configure the discussion categories as needed

### GitHub Pages Deployment

The `workflows/pages.yml` file configures automatic deployment of GitHub Pages when changes are pushed to the main branch. The workflow:

1. Checks out the repository
2. Sets up GitHub Pages
3. Uploads the contents of the `docs` directory as a GitHub Pages artifact
4. Deploys the artifact to GitHub Pages

## Customization

Feel free to modify these files to suit your specific needs. For example:

- Update the header images to match your project's branding
- Modify the discussion templates to include project-specific questions
- Add additional GitHub Actions workflows for CI/CD, testing, etc.
