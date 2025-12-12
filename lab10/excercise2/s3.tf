
# Defining random bucket suffix to ensure unique name
resource "random_id" "bucket_suffix" {
  count       = length(var.regions)
  byte_length = 8
}

# Creating buckets
resource "aws_s3_bucket" "s3_us_east_1" {
  bucket = "${var.bucket_name_prefix}-${var.regions[0]}-${random_id.bucket_suffix[0].hex}"
}

resource "aws_s3_bucket" "s3_us_west_2" {
  provider = aws.other_region
  bucket   = "${var.bucket_name_prefix}-${var.regions[1]}-${random_id.bucket_suffix[1].hex}"
}


# Setting versionng
resource "aws_s3_bucket_versioning" "s3_us_east_1" {
  bucket = aws_s3_bucket.s3_us_east_1.id

  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_versioning" "s3_us_west_2" {
  provider = aws.other_region
  bucket   = aws_s3_bucket.s3_us_west_2.id

  versioning_configuration {
    status = "Enabled"
  }
}


# Setting lifecycle configuration
resource "aws_s3_bucket_lifecycle_configuration" "s3_us_east_1" {
  bucket = aws_s3_bucket.s3_us_east_1.id

  rule {
    id     = "transition-to-glacier"
    status = "Enabled"
    filter {}

    transition {
      days          = var.lifecycle_days
      storage_class = var.lifecycle_storage_class
    }
  }
}


resource "aws_s3_bucket_lifecycle_configuration" "s3_us_west_2" {
  provider = aws.other_region
  bucket   = aws_s3_bucket.s3_us_west_2.id

  rule {
    id     = "transition-to-glacier"
    status = "Enabled"
    filter {}

    transition {
      days          = var.lifecycle_days
      storage_class = var.lifecycle_storage_class
    }
  }
}