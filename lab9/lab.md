# AWS LAB

## 1. Account setup

I used the aws-learner-lab account


## 2. S3

The model files are available on the s3 ![](./screenshots/s3.png)

I also added a download script to lab1 task [script](https://github.com/mnozkiewicz/MLOPS/blob/main/lab1/homework/download_models.py).

## 3. ECR

The docker image were pushed to a registry ![](./screenshots/ecr.png)

The latest image includes builts both for amd and arm architectures, so it could work both locally and linux/amd machines that are used in the fargate service.


## 4. Network setup


The final network schema looks like this

![](./screenshots/vpc1.png)
![](./screenshots/vpc2.png)

and also ALB

![](./screenshots/alb.png)


### 5. ECS and Fargate

![](./screenshots/ecs.png)


### 6. Running application


The ready application is accessible through load balancer.

![](./screenshots/app.png)