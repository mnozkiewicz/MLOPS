
variable "regions" {
  description = "List of AWS regions that the buckets are located in"
  type        = list(string)
  default     = ["us-east-1", "us-west-2"]
}

variable "bucket_name_prefix" {
  description = "Prefix for the multi-region bucket name"
  type        = string
  default     = "mlops-terraform-lab-bucket"
}

variable "lifecycle_days" {
  description = <<-EOT
    Lifecycle rule regarding infrequently accessed objects.
    Denotes the number of days before transitioning objects to S3 Glacier
  EOT
  type        = number
  default     = 90
}

variable "lifecycle_storage_class" {
  type    = string
  default = "GLACIER"
}