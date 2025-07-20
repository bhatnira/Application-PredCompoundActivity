# Deployment Instructions for Render.com

This document provides instructions for deploying the Compound Activity Prediction Suite to Render.com.

## Prerequisites

- A Render.com account
- Your application code pushed to a Git repository (GitHub, GitLab, or Bitbucket)

## Deployment Steps

### Option 1: Using the Render Dashboard (Recommended)

1. **Connect your repository:**
   - Go to [Render.com](https://render.com) and sign in
   - Click "New +" and select "Web Service"
   - Connect your Git repository containing this application

2. **Configure the service:**
   - **Name:** `compound-activity-prediction` (or any name you prefer)
   - **Environment:** Docker
   - **Plan:** Starter (or higher based on your needs)
   - **Branch:** main (or your default branch)

3. **Environment Variables (Optional):**
   The Dockerfile already sets these, but you can override them:
   ```
   STREAMLIT_SERVER_PORT=8080
   STREAMLIT_SERVER_ADDRESS=0.0.0.0
   STREAMLIT_SERVER_HEADLESS=true
   STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
   ```

4. **Deploy:**
   - Click "Create Web Service"
   - Render will automatically build and deploy your application
   - The build process takes about 5-10 minutes due to the ML dependencies

### Option 2: Using render.yaml (Infrastructure as Code)

1. Ensure the `render.yaml` file is in your repository root
2. Go to Render Dashboard → "New +" → "Blueprint"
3. Connect your repository and Render will use the `render.yaml` configuration

## Important Notes

### Resource Requirements
- **Memory:** The application requires at least 1GB RAM due to ML libraries (TensorFlow, scikit-learn, RDKit, etc.)
- **Build Time:** Initial deployment takes 5-10 minutes to install dependencies
- **Storage:** About 2-3GB for all dependencies

### File Upload Limitations
- Render.com has file upload size limits
- For large datasets, consider using cloud storage (AWS S3, Google Cloud Storage) and modify the app to accept URLs

### Cost Optimization
- **Starter Plan:** Good for testing and light usage
- **Standard Plan:** Recommended for production with consistent traffic
- **Pro Plan:** For high-traffic applications

## Monitoring and Troubleshooting

### Health Check
The application includes a health check endpoint at `/_stcore/health`

### Logs
- View logs in the Render dashboard under your service → "Logs"
- Common issues:
  - Memory issues: Upgrade to a higher plan
  - Timeout during build: Dependencies are being installed
  - Import errors: Missing dependencies in requirements.txt

### Performance Tips
1. **Cold Start:** First request after inactivity may be slow
2. **Memory Usage:** Monitor memory usage for large datasets
3. **Caching:** The app uses Streamlit's caching for better performance

## Local Testing

Before deploying, test the Docker container locally:

```bash
# Build the image
docker build -t compound-activity-prediction .

# Run the container
docker run -p 8080:8080 compound-activity-prediction

# Open http://localhost:8080 in your browser
```

## Security Considerations

- The application runs as a non-root user for security
- CORS is disabled for the containerized environment
- Consider adding authentication for production use

## Support

If you encounter issues:
1. Check the Render service logs
2. Verify all files are committed to your repository
3. Ensure the Dockerfile and requirements.txt are correct
4. Contact Render support for platform-specific issues
