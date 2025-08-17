# Email Notification Setup for IRIS Pipeline

## Required GitHub Secrets

To enable email notifications for the IRIS Pipeline evaluation reports, you need to set up the following secrets in your GitHub repository:

### 1. Navigate to Repository Settings
- Go to your repository: https://github.com/techy-bytes/iris-pipeline
- Click on "Settings" tab
- In the left sidebar, click "Secrets and variables" → "Actions"

### 2. Add Required Secrets

Click "New repository secret" and add each of the following:

#### EMAIL_USERNAME
- **Name**: `EMAIL_USERNAME`
- **Value**: Your Gmail address (e.g., `your-email@gmail.com`)

#### EMAIL_PASSWORD
- **Name**: `EMAIL_PASSWORD`
- **Value**: Your Gmail App Password (NOT your regular password)
- **How to get App Password**:
  1. Go to your Google Account settings
  2. Enable 2-Factor Authentication if not already enabled
  3. Go to Security → 2-Step Verification → App passwords
  4. Generate an app password for "Mail"
  5. Use this 16-character password as the secret value

#### EMAIL_TO
- **Name**: `EMAIL_TO`
- **Value**: Recipient email address (can be the same as EMAIL_USERNAME)

### 3. Alternative Email Providers

If you don't want to use Gmail, you can modify the workflow to use other providers:

#### Microsoft Outlook/Hotmail
```yaml
server_address: smtp-mail.outlook.com
server_port: 587
```

#### Yahoo Mail
```yaml
server_address: smtp.mail.yahoo.com
server_port: 587
```

#### Custom SMTP Server
```yaml
server_address: your-smtp-server.com
server_port: 587  # or 465 for SSL
```

## How It Works

### For Pull Requests
- CML posts evaluation report as a comment on the PR
- Email is sent with full report and attachments

### For Direct Pushes (like dev branch)
- Creates a GitHub Issue with the evaluation report
- Email is sent with full report and attachments
- Artifacts are uploaded for 30 days

## Email Content

The email includes:
- **Subject**: Branch name and workflow run number
- **Body**: Summary with links to full results
- **Attachments**:
  - `report.md` - Detailed markdown report
  - `gemma_evaluation_results.json` - Raw evaluation data
  - `gemma_metrics.csv` - Metrics in CSV format

## Troubleshooting

### Email Not Received
1. Check GitHub Actions logs for email step
2. Verify all secrets are correctly set
3. Check spam/junk folder
4. Ensure Gmail App Password is correct

### Authentication Errors
- Make sure 2FA is enabled on Gmail
- Use App Password, not regular password
- Check that EMAIL_USERNAME is the full email address

### SMTP Errors
- Verify server address and port
- Some networks block SMTP ports (try different network)
- Check if your email provider requires specific settings

## Testing

After setting up secrets, push any change to test:

```bash
# Make a small change and push
echo "# Test change" >> README.md
git add README.md
git commit -m "test: email notification setup"
git push origin dev
```

You should receive an email with the evaluation report within a few minutes.
