
variable "github_token" {
  description = "GitHub Personal Access Token"
  type        = string
  sensitive   = true # Marks this variable as sensitive, preventing it from appearing in logs and console output
}

variable "repository_name" {
  description = "terraform-managed-repo"
  type        = string
  default     = "MLOPS-terraform-lab"
}

variable "repository_description" {
  description = "Repository containing terraform code for a MLOPS course task."
  type        = string
  default     = "Repository managed by Terraform"
}

variable "publicly_visible" {
  description = "Whether the GitHub repository should be public"
  type        = bool
  default     = false
}