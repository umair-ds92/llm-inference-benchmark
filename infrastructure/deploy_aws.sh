#!/bin/bash
# AWS Infrastructure Deployment Script for LLM Inference Benchmarking
# This script sets up the necessary AWS resources for running benchmarks

set -e  # Exit on error

# Default values
INSTANCE_TYPE="${INSTANCE_TYPE:-g5.xlarge}"
REGION="${REGION:-us-east-1}"
KEY_NAME="${KEY_NAME:-llm-benchmark-key}"
SECURITY_GROUP_NAME="llm-benchmark-sg"
VOLUME_SIZE="${VOLUME_SIZE:-100}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if AWS CLI is installed
if ! command -v aws &> /dev/null; then
    log_error "AWS CLI not found. Please install it first:"
    log_error "https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
    exit 1
fi

# Check if AWS credentials are configured
if ! aws sts get-caller-identity &> /dev/null; then
    log_error "AWS credentials not configured. Run 'aws configure' first."
    exit 1
fi

log_info "Starting AWS infrastructure deployment..."
log_info "Region: $REGION"
log_info "Instance Type: $INSTANCE_TYPE"

# Create security group if it doesn't exist
log_info "Checking security group..."
SG_ID=$(aws ec2 describe-security-groups \
    --region $REGION \
    --filters Name=group-name,Values=$SECURITY_GROUP_NAME \
    --query 'SecurityGroups[0].GroupId' \
    --output text 2>/dev/null)

if [ "$SG_ID" == "None" ] || [ -z "$SG_ID" ]; then
    log_info "Creating security group: $SECURITY_GROUP_NAME"
    SG_ID=$(aws ec2 create-security-group \
        --region $REGION \
        --group-name $SECURITY_GROUP_NAME \
        --description "Security group for LLM benchmarking instances" \
        --query 'GroupId' \
        --output text)
    
    # Add SSH access
    aws ec2 authorize-security-group-ingress \
        --region $REGION \
        --group-id $SG_ID \
        --protocol tcp \
        --port 22 \
        --cidr 0.0.0.0/0
    
    # Add custom ports for monitoring (optional)
    aws ec2 authorize-security-group-ingress \
        --region $REGION \
        --group-id $SG_ID \
        --protocol tcp \
        --port 8000-8100 \
        --cidr 0.0.0.0/0
    
    log_info "Security group created: $SG_ID"
else
    log_info "Security group already exists: $SG_ID"
fi

# Check if key pair exists
log_info "Checking SSH key pair..."
if ! aws ec2 describe-key-pairs --region $REGION --key-names $KEY_NAME &> /dev/null; then
    log_warn "Key pair '$KEY_NAME' not found."
    log_info "Creating new key pair..."
    aws ec2 create-key-pair \
        --region $REGION \
        --key-name $KEY_NAME \
        --query 'KeyMaterial' \
        --output text > ~/.ssh/${KEY_NAME}.pem
    chmod 400 ~/.ssh/${KEY_NAME}.pem
    log_info "Key pair created and saved to ~/.ssh/${KEY_NAME}.pem"
else
    log_info "Key pair already exists: $KEY_NAME"
fi

# Get the latest Deep Learning AMI (Ubuntu) with CUDA
log_info "Finding latest Deep Learning AMI..."
AMI_ID=$(aws ec2 describe-images \
    --region $REGION \
    --owners amazon \
    --filters \
        "Name=name,Values=Deep Learning AMI GPU PyTorch * (Ubuntu 20.04)*" \
        "Name=state,Values=available" \
    --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
    --output text)

if [ -z "$AMI_ID" ] || [ "$AMI_ID" == "None" ]; then
    log_error "Could not find suitable AMI. Using default Ubuntu 20.04 AMI."
    # Fallback to standard Ubuntu AMI
    AMI_ID=$(aws ec2 describe-images \
        --region $REGION \
        --owners 099720109477 \
        --filters \
            "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-focal-20.04-amd64-server-*" \
            "Name=state,Values=available" \
        --query 'sort_by(Images, &CreationDate)[-1].ImageId' \
        --output text)
fi

log_info "Using AMI: $AMI_ID"

# Create user data script for instance initialization
cat > /tmp/user_data.sh << 'EOF'
#!/bin/bash
# User data script for EC2 instance initialization

# Update system
apt-get update
apt-get install -y python3-pip git htop nvtop

# Install NVIDIA drivers if not present
if ! command -v nvidia-smi &> /dev/null; then
    echo "Installing NVIDIA drivers..."
    apt-get install -y nvidia-driver-525 nvidia-utils-525
fi

# Install Docker
if ! command -v docker &> /dev/null; then
    echo "Installing Docker..."
    curl -fsSL https://get.docker.com -o get-docker.sh
    sh get-docker.sh
    usermod -aG docker ubuntu
fi

# Create workspace
mkdir -p /workspace
chown ubuntu:ubuntu /workspace

echo "Instance initialization complete"
EOF

# Launch EC2 instance
log_info "Launching EC2 instance..."
INSTANCE_ID=$(aws ec2 run-instances \
    --region $REGION \
    --image-id $AMI_ID \
    --instance-type $INSTANCE_TYPE \
    --key-name $KEY_NAME \
    --security-group-ids $SG_ID \
    --block-device-mappings "[{\"DeviceName\":\"/dev/sda1\",\"Ebs\":{\"VolumeSize\":$VOLUME_SIZE,\"VolumeType\":\"gp3\"}}]" \
    --user-data file:///tmp/user_data.sh \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=llm-benchmark},{Key=Project,Value=llm-inference-benchmarking}]" \
    --query 'Instances[0].InstanceId' \
    --output text)

log_info "Instance launched: $INSTANCE_ID"
log_info "Waiting for instance to start..."

aws ec2 wait instance-running --region $REGION --instance-ids $INSTANCE_ID

# Get instance public IP
PUBLIC_IP=$(aws ec2 describe-instances \
    --region $REGION \
    --instance-ids $INSTANCE_ID \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

log_info "Instance is running!"
log_info "Instance ID: $INSTANCE_ID"
log_info "Public IP: $PUBLIC_IP"
log_info ""
log_info "To connect to the instance:"
log_info "  ssh -i ~/.ssh/${KEY_NAME}.pem ubuntu@${PUBLIC_IP}"
log_info ""
log_info "To stop the instance:"
log_info "  aws ec2 stop-instances --region $REGION --instance-ids $INSTANCE_ID"
log_info ""
log_info "To terminate the instance:"
log_info "  aws ec2 terminate-instances --region $REGION --instance-ids $INSTANCE_ID"

# Save instance info to file
cat > instance_info.txt << EOF
Instance ID: $INSTANCE_ID
Public IP: $PUBLIC_IP
Region: $REGION
Instance Type: $INSTANCE_TYPE
Key Name: $KEY_NAME
Security Group: $SG_ID
AMI: $AMI_ID
EOF

log_info "Instance information saved to instance_info.txt"

# Cleanup
rm /tmp/user_data.sh

log_info "Deployment complete!"