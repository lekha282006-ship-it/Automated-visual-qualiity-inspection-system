"""Industrial network clients with optional real library support.

This module provides `create_client(protocol, endpoint, **kwargs)` which will attempt
to create a real client using `pymodbus` for Modbus TCP or `opcua` (python-opcua) for OPC-UA.
If the libraries are not available, it falls back to safe stubs with identical interfaces.
"""
import time
from typing import Optional, Dict, Any


# --- Modbus implementation ---
try:
    # pymodbus v2+ imports
    from pymodbus.client.sync import ModbusTcpClient as _ModbusTcpClient  # type: ignore
    _HAS_PYMODBUS = True
except Exception:
    try:
        # newer pymodbus structure fallback
        from pymodbus.client import ModbusTcpClient as _ModbusTcpClient  # type: ignore
        _HAS_PYMODBUS = True
    except Exception:
        _HAS_PYMODBUS = False


class ModbusClientStub:
    def __init__(self, endpoint: Optional[str] = None):
        self.endpoint = endpoint or "127.0.0.1"
        self.port = 502
        if isinstance(self.endpoint, str) and ":" in self.endpoint:
            parts = self.endpoint.split(":")
            self.endpoint = parts[0]
            try:
                self.port = int(parts[1])
            except Exception:
                self.port = 502
        self.connected = False
        self._client = None

    def connect(self):
        print(f"[MODBUS] (stub) connecting to {self.endpoint}:{self.port}")
        time.sleep(0.05)
        self.connected = True

    def write_coil(self, address: int, value: bool):
        if not self.connected:
            self.connect()
        print(f"[MODBUS] (stub) write_coil {address} -> {value}")

    def close(self):
        print("[MODBUS] (stub) close")
        self.connected = False

    def send_reject(self, signal: Dict[str, Any]):
        self.write_coil(1, True)
        time.sleep(0.1)
        self.write_coil(1, False)


class ModbusClientReal(ModbusClientStub):
    def __init__(self, endpoint: Optional[str] = None):
        super().__init__(endpoint)
        self._client = None

    def connect(self):
        if not _HAS_PYMODBUS:
            return super().connect()
        try:
            self._client = _ModbusTcpClient(self.endpoint, port=self.port)
            self._client.connect()
            self.connected = True
            print(f"[MODBUS] connected to {self.endpoint}:{self.port}")
        except Exception as e:
            print("[MODBUS] connection failed:", e)
            self.connected = False

    def write_coil(self, address: int, value: bool):
        if not self.connected:
            self.connect()
        if self.connected and _HAS_PYMODBUS and self._client:
            try:
                self._client.write_coil(address, int(value))
            except Exception as e:
                print("[MODBUS] write_coil failed:", e)
        else:
            super().write_coil(address, value)

    def close(self):
        try:
            if self._client:
                self._client.close()
        except Exception:
            pass
        self.connected = False


# --- OPC-UA implementation ---
try:
    from opcua import Client as _OPCUAClient  # type: ignore
    from opcua import ua as _ua  # type: ignore
    _HAS_OPCUA = True
except Exception:
    _HAS_OPCUA = False


class OPCUAClientStub:
    def __init__(self, endpoint: Optional[str] = None):
        self.endpoint = endpoint or "opc.tcp://127.0.0.1:4840"
        self.connected = False
        self._client = None

    def connect(self):
        print(f"[OPCUA] (stub) connecting to {self.endpoint}")
        time.sleep(0.05)
        self.connected = True

    def write_node(self, node_id: str, value: Any):
        if not self.connected:
            self.connect()
        print(f"[OPCUA] (stub) write_node {node_id} -> {value}")

    def close(self):
        print("[OPCUA] (stub) close")
        self.connected = False

    def send_reject(self, signal: Dict[str, Any]):
        self.write_node("ns=2;s=Reject", True)
        self.write_node("ns=2;s=RejectTS", float(signal.get("timestamp", 0.0)))


class OPCUAClientReal(OPCUAClientStub):
    def __init__(self, endpoint: Optional[str] = None, username: Optional[str] = None, password: Optional[str] = None):
        super().__init__(endpoint)
        self.username = username
        self.password = password

    def connect(self):
        if not _HAS_OPCUA:
            return super().connect()
        try:
            self._client = _OPCUAClient(self.endpoint)
            if self.username and self.password:
                self._client.set_user(self.username)
                self._client.set_password(self.password)
            self._client.connect()
            self.connected = True
            print(f"[OPCUA] connected to {self.endpoint}")
        except Exception as e:
            print("[OPCUA] connection failed:", e)
            self.connected = False

    def write_node(self, node_id: str, value: Any):
        if not self.connected:
            self.connect()
        if self.connected and _HAS_OPCUA and self._client:
            try:
                node = self._client.get_node(node_id)
                # adapt value types
                node.set_value(value)
            except Exception as e:
                print("[OPCUA] write_node failed:", e)
        else:
            super().write_node(node_id, value)

    def close(self):
        try:
            if self._client:
                self._client.disconnect()
        except Exception:
            pass
        self.connected = False


def create_client(protocol: str, endpoint: Optional[str] = None, **kwargs):
    p = (protocol or "").lower()
    if p in ("modbus", "modbus-tcp", "modbus_tcp"):
        # try real client if available
        if _HAS_PYMODBUS:
            return ModbusClientReal(endpoint)
        return ModbusClientStub(endpoint)
    if p in ("opcua", "opc-ua", "opc_tcp"):
        if _HAS_OPCUA:
            return OPCUAClientReal(endpoint, username=kwargs.get('username'), password=kwargs.get('password'))
        return OPCUAClientStub(endpoint)
    raise ValueError(f"Unknown protocol: {protocol}")
