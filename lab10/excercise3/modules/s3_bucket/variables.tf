variable "bucket_name_prefix" {
  description = "Prefix for the multi-region bucket name"
  type        = string
}

variable "region" {
  description = "AWS region where this S3 bucket will be deployed"
  type        = string
}

variable "random_suffix" {
  description = "Random suffix to ensure bucket name is unique"
  type        = string
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
  description = "Storage class to transition objects to after lifecycle_days"
  type        = string
  default     = "GLACIER"
}
