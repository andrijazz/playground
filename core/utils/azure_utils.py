import os

import azure.storage.file as azure
from azure.storage.blob import BlobServiceClient

from utils.logger import get_logger

logger = get_logger()


class AzureUtils:
    def __init__(self, account_name, azure_api_key):
        self.account_name = account_name
        self.azure_api_key = azure_api_key
        self.connect_str = "DefaultEndpointsProtocol=https;AccountName={};AccountKey={};EndpointSuffix=core.windows.net".format(self.account_name, self.azure_api_key)

    def upload_file(self, f, share_name, dest_dir, create_dest_dir=False):
        """
        Uploads file to azure dest

        :param f:
        :param share_name:
        :param dest_dir:
        :param create_dest_dir:
        :return: success flag
        """
        try:
            file_service = azure.FileService(account_name=self.account_name, account_key=self.azure_api_key)

            # create_target_path
            if create_dest_dir:
                folder_status = file_service.create_directory(share_name, dest_dir)
                if not folder_status:
                    raise Exception('Failed to create new {} share folder.'.format(dest_dir))

            dest_filename = os.path.basename(f)
            file_service.create_file_from_path(share_name, dest_dir, dest_filename, f, max_connections=4)
            azure_file_path = os.path.join('https://trendage.file.core.windows.net', share_name, dest_dir, dest_filename)
            return azure_file_path
        except Exception as e:
            logger.error('Failed to upload file {} because of exception: {}.'.format(f, str(e)))
        return None

    def download_file(self, file_name, share_name, azure_dir, output_file_path):
        try:
            file_service = azure.FileService(account_name=self.account_name, account_key=self.azure_api_key)
            file_service.get_file_to_path(share_name, azure_dir, file_name, output_file_path)
            return True
        except Exception as e:
            logger.error('Failed to download file {} because of exception: {}.'.format(file_name, str(e)))
        return False

    def upload_file_to_blob(self, upload_file_path, dst_filename, container='fakes'):
        try:
            blob_service_client = BlobServiceClient.from_connection_string(self.connect_str)

            # Create a blob client using the local file name as the name for the blob
            blob_client = blob_service_client.get_blob_client(container=container, blob=dst_filename)

            logger.info('Uploading {} to Azure Storage as blob'.format(dst_filename))

            # Upload the created file
            with open(upload_file_path, "rb") as data:
                res = blob_client.upload_blob(data, overwrite=True)
                logger.info('Uploading {} to Azure Storage completed. res = {}'.format(dst_filename, res))

        except Exception as e:
            logger.error('Failed to upload file {} because of exception: {}.'.format(upload_file_path, str(e)))
