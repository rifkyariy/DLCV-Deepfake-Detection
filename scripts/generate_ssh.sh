#!/bin/bash

# Clear terminal for readability
clear

echo "========================================================"
echo "   SSH Key Generator for RunPod (and other services)"
echo "========================================================"
echo ""

# 1. Ask for an email/comment to identify the key
read -p "Enter your email address (for key comment): " user_email

if [ -z "$user_email" ]; then
    echo "No email provided. Using default comment 'runpod-key'"
    user_email="runpod-key"
fi

# 2. Define key path (using Ed25519 for better security/performance)
KEY_NAME="id_ed25519_runpod"
KEY_PATH="$HOME/.ssh/$KEY_NAME"

echo ""
echo "Generating SSH key pair at: $KEY_PATH"

# 3. Generate the key
# -t ed25519: Specifies the type of key
# -C: Adds a comment (your email)
# -f: Specifies the filename
# -N "": Creates it with NO passphrase (press Enter automatically) for easier automation. 
#        Remove -N "" if you want to manually set a passphrase for extra security.
ssh-keygen -t ed25519 -C "$user_email" -f "$KEY_PATH" -N ""

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Success! Key generated."
    echo ""
    echo "--------------------------------------------------------"
    echo "COPY THE TEXT BELOW (This is your Public Key):"
    echo "--------------------------------------------------------"
    echo ""
    
    # 4. Display the public key
    cat "$KEY_PATH.pub"
    
    echo ""
    echo "--------------------------------------------------------"
    echo "Paste the key above into your RunPod account settings."
    echo "========================================================"
else
    echo "❌ Error: Something went wrong generating the key."
fi