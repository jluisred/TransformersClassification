#
# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#

from io import BytesIO
import boto3


def get_s3_resource(access_key_id="", secret_access_key="", region="us-east-1"):
    return boto3.resource(
        "s3",
        region_name=region,
        aws_access_key_id=access_key_id,
        aws_secret_access_key=secret_access_key,
    )


def save_to_s3(byte_obj, s3_resource, bucket_name, key):
    s3_resource.Object(bucket_name, key).put(Body=byte_obj)


def load_from_s3(s3_resource, bucket_name, key):
    obj = s3_resource.Object(bucket_name, key)
    return BytesIO(obj.get()["Body"].read())


def list_in_s3(s3_resource, bucket_name, prefix=""):
    bucket = s3_resource.Bucket(bucket_name)
    filename_position = len(prefix) + 1 if prefix else 0
    return [obj.key[filename_position:] for obj in bucket.objects.filter(Prefix=prefix)]