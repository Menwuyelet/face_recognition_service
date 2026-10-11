"""
face_recognition_service.crypto
===============================

Fernet-based symmetric encryption / decryption for face embeddings.

Embeddings are serialized as raw ``float32`` bytes before encryption,
making the ciphertext safe to store in databases or transmit over the
network.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from cryptography.fernet import Fernet, InvalidToken

logger = logging.getLogger(__name__)


class EmbeddingCrypto:
    """Encrypt and decrypt face embeddings using Fernet symmetric encryption.

    Parameters
    ----------
    key : str or bytes or None
        A Fernet key (URL-safe base64-encoded 32-byte key).  If ``None``,
        a fresh key is generated and exposed via :attr:`key`.

    Examples
    --------
    ::

        crypto = EmbeddingCrypto()           # auto-generates key
        print(crypto.key)                    # save this key!

        encrypted = crypto.encrypt(embedding)
        restored  = crypto.decrypt(encrypted)
    """

    def __init__(self, key: Optional[str | bytes] = None) -> None:
        if key is None:
            self._key = Fernet.generate_key()
            logger.info("CRYPTO_KEY_GENERATED | (new key auto-generated)")
        else:
            self._key = key if isinstance(key, bytes) else key.encode()

        self._fernet = Fernet(self._key)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def key(self) -> str:
        """Return the Fernet key as a UTF-8 string (for storage)."""
        return self._key.decode() if isinstance(self._key, bytes) else self._key

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encrypt(self, embedding: np.ndarray) -> bytes:
        """Serialize and encrypt a face embedding.

        Parameters
        ----------
        embedding : numpy.ndarray
            A float32 embedding vector (typically 512-d).

        Returns
        -------
        bytes
            Fernet-encrypted ciphertext containing the raw embedding bytes.
        """
        raw = np.asarray(embedding, dtype=np.float32).tobytes()

        logger.info(
            "CRYPTO_ENCRYPT_STARTED | raw_size=%d",
            len(raw),
        )

        encrypted = self._fernet.encrypt(raw)

        logger.info(
            "CRYPTO_ENCRYPT_SUCCESS | encrypted_size=%d",
            len(encrypted),
        )

        return encrypted

    def decrypt(self, ciphertext: bytes) -> np.ndarray:
        """Decrypt ciphertext back to a numpy embedding array.

        Parameters
        ----------
        ciphertext : bytes
            Encrypted payload produced by :meth:`encrypt`.

        Returns
        -------
        numpy.ndarray
            The restored embedding vector (dtype ``float32``).

        Raises
        ------
        cryptography.fernet.InvalidToken
            If the ciphertext is invalid or the key is wrong.
        """
        logger.info(
            "CRYPTO_DECRYPT_STARTED | ciphertext_size=%d",
            len(ciphertext),
        )

        raw = self._fernet.decrypt(ciphertext)
        embedding = np.frombuffer(raw, dtype=np.float32).copy()

        logger.info(
            "CRYPTO_DECRYPT_SUCCESS | embedding_dim=%d",
            embedding.shape[0],
        )

        return embedding

    # ------------------------------------------------------------------
    # Key generation utility
    # ------------------------------------------------------------------

    @staticmethod
    def generate_key() -> str:
        """Generate a new Fernet key.

        Returns
        -------
        str
            A URL-safe base64-encoded 32-byte key suitable for passing
            to the constructor.
        """
        return Fernet.generate_key().decode()
