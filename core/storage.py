"""
Storage backend for Centrifuge.
Provides MinIO S3 storage with local filesystem fallback.
"""

import os
import logging
import hashlib
import asyncio
from typing import Optional, Dict, Any, BinaryIO
from abc import ABC, abstractmethod
import json
import aiofiles
from datetime import datetime, timedelta

try:
    from minio import Minio
    from minio.error import S3Error
    HAS_MINIO = True
except ImportError:
    HAS_MINIO = False
    
logger = logging.getLogger(__name__)


class StorageBackend(ABC):
    """Abstract storage backend interface."""
    
    @abstractmethod
    async def upload(self, source_path: str, dest_path: str) -> str:
        """Upload file to storage."""
        pass
    
    @abstractmethod
    async def download(self, source_path: str, dest_path: str) -> None:
        """Download file from storage."""
        pass
    
    @abstractmethod
    async def exists(self, path: str) -> bool:
        """Check if file exists."""
        pass
    
    @abstractmethod
    async def delete(self, path: str) -> None:
        """Delete file from storage."""
        pass
    
    @abstractmethod
    async def get_url(self, path: str, expires: int = 3600) -> str:
        """Get presigned URL for file."""
        pass
    
    @abstractmethod
    async def list_files(self, prefix: str) -> list:
        """List files with prefix."""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """Close storage connections."""
        pass


class MinIOStorage(StorageBackend):
    """MinIO S3-compatible storage backend."""
    
    def __init__(self, 
                 endpoint: str = None,
                 access_key: str = None,
                 secret_key: str = None,
                 bucket: str = None,
                 secure: bool = False):
        """
        Initialize MinIO storage.
        
        Args:
            endpoint: MinIO endpoint (default from env MINIO_ENDPOINT)
            access_key: Access key (default from env MINIO_ACCESS_KEY)
            secret_key: Secret key (default from env MINIO_SECRET_KEY)
            bucket: Bucket name (default from env MINIO_BUCKET or 'centrifuge')
            secure: Use HTTPS
        """
        if not HAS_MINIO:
            raise ImportError("minio package not installed")
        
        # get config from env or defaults
        self.endpoint = endpoint or os.getenv('MINIO_ENDPOINT', 'minio:9000')
        self.access_key = access_key or os.getenv('MINIO_ACCESS_KEY', 'minioadmin')
        self.secret_key = secret_key or os.getenv('MINIO_SECRET_KEY', 'minioadmin')
        self.bucket = bucket or os.getenv('MINIO_BUCKET', 'centrifuge')
        self.secure = secure
        
        # initialize client
        self.client = Minio(
            self.endpoint,
            access_key=self.access_key,
            secret_key=self.secret_key,
            secure=self.secure
        )
        
        # ensure bucket exists
        self._ensure_bucket()
        
        logger.info(f"MinIO storage initialized: {self.endpoint}/{self.bucket}")
    
    def _ensure_bucket(self):
        """Ensure bucket exists."""
        try:
            if not self.client.bucket_exists(self.bucket):
                self.client.make_bucket(self.bucket)
                logger.info(f"Created bucket: {self.bucket}")
        except Exception as e:
            logger.error(f"Failed to ensure bucket: {e}")
            raise
    
    async def upload(self, source_path: str, dest_path: str) -> str:
        """
        Upload file to MinIO.
        
        Args:
            source_path: Local file path
            dest_path: Destination path in bucket
            
        Returns:
            Object path
        """
        try:
            # get file size
            file_size = os.path.getsize(source_path)
            
            # upload with retry
            for attempt in range(3):
                try:
                    result = await asyncio.to_thread(
                        self.client.fput_object,
                        self.bucket,
                        dest_path,
                        source_path
                    )
                    
                    logger.info(f"Uploaded {source_path} to {dest_path}")
                    return f"s3://{self.bucket}/{dest_path}"
                    
                except S3Error as e:
                    if attempt < 2:
                        wait_time = 2 ** attempt
                        logger.warning(f"Upload retry {attempt + 1} after {wait_time}s: {e}")
                        await asyncio.sleep(wait_time)
                    else:
                        raise
                        
        except Exception as e:
            logger.error(f"Failed to upload {source_path}: {e}")
            raise
    
    async def download(self, source_path: str, dest_path: str) -> None:
        """
        Download file from MinIO.
        
        Args:
            source_path: Source path in bucket
            dest_path: Local destination path
        """
        try:
            # strip s3:// prefix if present
            if source_path.startswith('s3://'):
                parts = source_path[5:].split('/', 1)
                if len(parts) == 2:
                    source_path = parts[1]
            
            # download with retry
            for attempt in range(3):
                try:
                    await asyncio.to_thread(
                        self.client.fget_object,
                        self.bucket,
                        source_path,
                        dest_path
                    )
                    
                    logger.info(f"Downloaded {source_path} to {dest_path}")
                    return
                    
                except S3Error as e:
                    if attempt < 2:
                        wait_time = 2 ** attempt
                        logger.warning(f"Download retry {attempt + 1} after {wait_time}s: {e}")
                        await asyncio.sleep(wait_time)
                    else:
                        raise
                        
        except Exception as e:
            logger.error(f"Failed to download {source_path}: {e}")
            raise
    
    async def exists(self, path: str) -> bool:
        """
        Check if file exists in MinIO.
        
        Args:
            path: File path in bucket
            
        Returns:
            True if exists
        """
        try:
            await asyncio.to_thread(
                self.client.stat_object,
                self.bucket,
                path
            )
            return True
        except S3Error:
            return False
    
    async def delete(self, path: str) -> None:
        """
        Delete file from MinIO.
        
        Args:
            path: File path in bucket
        """
        try:
            await asyncio.to_thread(
                self.client.remove_object,
                self.bucket,
                path
            )
            logger.info(f"Deleted {path}")
        except Exception as e:
            logger.error(f"Failed to delete {path}: {e}")
            raise
    
    async def get_url(self, path: str, expires: int = 3600) -> str:
        """
        Get presigned URL for file.
        
        Args:
            path: File path in bucket
            expires: URL expiration in seconds
            
        Returns:
            Presigned URL
        """
        try:
            url = await asyncio.to_thread(
                self.client.presigned_get_object,
                self.bucket,
                path,
                expires=timedelta(seconds=expires)
            )
            return url
        except Exception as e:
            logger.error(f"Failed to get URL for {path}: {e}")
            raise
    
    async def list_files(self, prefix: str) -> list:
        """
        List files with prefix.
        
        Args:
            prefix: Path prefix
            
        Returns:
            List of file paths
        """
        try:
            objects = await asyncio.to_thread(
                lambda: list(self.client.list_objects(
                    self.bucket,
                    prefix=prefix,
                    recursive=True
                ))
            )
            return [obj.object_name for obj in objects]
        except Exception as e:
            logger.error(f"Failed to list files with prefix {prefix}: {e}")
            raise
    
    async def close(self) -> None:
        """Close MinIO connection."""
        # MinIO client doesn't need explicit closing
        pass


class LocalStorage(StorageBackend):
    """Local filesystem storage backend."""
    
    def __init__(self, base_path: str = None):
        """
        Initialize local storage.
        
        Args:
            base_path: Base directory for storage (default /tmp/centrifuge)
        """
        self.base_path = base_path or os.getenv('LOCAL_STORAGE_PATH', '/tmp/centrifuge')
        
        # ensure base path exists
        os.makedirs(self.base_path, exist_ok=True)
        
        logger.info(f"Local storage initialized: {self.base_path}")
    
    def _full_path(self, path: str) -> str:
        """Get full local path."""
        # strip file:// prefix if present
        if path.startswith('file://'):
            path = path[7:]
        
        # ensure relative to base path
        if os.path.isabs(path):
            path = path.lstrip('/')
        
        return os.path.join(self.base_path, path)
    
    async def upload(self, source_path: str, dest_path: str) -> str:
        """
        Copy file to local storage.
        
        Args:
            source_path: Source file path
            dest_path: Destination path
            
        Returns:
            File path
        """
        try:
            full_dest = self._full_path(dest_path)
            
            # ensure directory exists
            os.makedirs(os.path.dirname(full_dest), exist_ok=True)
            
            # copy file
            async with aiofiles.open(source_path, 'rb') as src:
                async with aiofiles.open(full_dest, 'wb') as dst:
                    await dst.write(await src.read())
            
            logger.info(f"Copied {source_path} to {full_dest}")
            return f"file://{full_dest}"
            
        except Exception as e:
            logger.error(f"Failed to upload {source_path}: {e}")
            raise
    
    async def download(self, source_path: str, dest_path: str) -> None:
        """
        Copy file from local storage.
        
        Args:
            source_path: Source path
            dest_path: Destination file path
        """
        try:
            full_source = self._full_path(source_path)
            
            # ensure destination directory exists
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            
            # copy file
            async with aiofiles.open(full_source, 'rb') as src:
                async with aiofiles.open(dest_path, 'wb') as dst:
                    await dst.write(await src.read())
            
            logger.info(f"Copied {full_source} to {dest_path}")
            
        except Exception as e:
            logger.error(f"Failed to download {source_path}: {e}")
            raise
    
    async def exists(self, path: str) -> bool:
        """
        Check if file exists.
        
        Args:
            path: File path
            
        Returns:
            True if exists
        """
        full_path = self._full_path(path)
        return os.path.exists(full_path)
    
    async def delete(self, path: str) -> None:
        """
        Delete file.
        
        Args:
            path: File path
        """
        try:
            full_path = self._full_path(path)
            if os.path.exists(full_path):
                os.remove(full_path)
                logger.info(f"Deleted {full_path}")
        except Exception as e:
            logger.error(f"Failed to delete {path}: {e}")
            raise
    
    async def get_url(self, path: str, expires: int = 3600) -> str:
        """
        Get file URL (just returns local path).
        
        Args:
            path: File path
            expires: Ignored for local storage
            
        Returns:
            File URL
        """
        full_path = self._full_path(path)
        return f"file://{full_path}"
    
    async def list_files(self, prefix: str) -> list:
        """
        List files with prefix.
        
        Args:
            prefix: Path prefix
            
        Returns:
            List of file paths
        """
        try:
            full_prefix = self._full_path(prefix)
            files = []
            
            for root, _, filenames in os.walk(full_prefix):
                for filename in filenames:
                    full = os.path.join(root, filename)
                    relative = os.path.relpath(full, self.base_path)
                    files.append(relative)
            
            return files
            
        except Exception as e:
            logger.error(f"Failed to list files with prefix {prefix}: {e}")
            raise
    
    async def close(self) -> None:
        """Close storage (no-op for local)."""
        pass


# global storage instance
_storage: Optional[StorageBackend] = None


async def get_storage_backend(use_local: bool = None) -> StorageBackend:
    """
    Get storage backend instance.
    
    Args:
        use_local: Force local storage (default from env USE_LOCAL_STORAGE)
        
    Returns:
        Storage backend
    """
    global _storage
    
    if not _storage:
        # check if we should use local storage
        if use_local is None:
            use_local = os.getenv('USE_LOCAL_STORAGE', 'false').lower() == 'true'
        
        if use_local or not HAS_MINIO:
            logger.info("Using local storage backend")
            _storage = LocalStorage()
        else:
            try:
                logger.info("Using MinIO storage backend")
                _storage = MinIOStorage()
            except Exception as e:
                logger.warning(f"Failed to initialize MinIO, falling back to local: {e}")
                _storage = LocalStorage()
    
    return _storage


async def close_storage() -> None:
    """Close storage backend."""
    global _storage
    
    if _storage:
        await _storage.close()
        _storage = None