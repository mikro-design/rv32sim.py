import socket


def safe_close(sock):
    if not sock:
        return
    try:
        sock.close()
    except Exception:
        pass


def run_server_loop(
    host,
    port,
    running_check,
    set_running,
    on_client,
    messages,
    client_setter=None,
    server_setter=None,
    configure_client=None,
    backlog=1,
    timeout=1.0,
):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        server.bind((host, port))
    except OSError as e:
        print(messages["bind_failed"].format(error=e))
        set_running(False)
        return
    server.listen(backlog)
    if server_setter:
        server_setter(server)
    while running_check():
        try:
            server.settimeout(timeout)
            client, _ = server.accept()
            if configure_client:
                configure_client(client)
            print(messages["client_connected"])
            if client_setter:
                client_setter(client)
            try:
                on_client(client)
            finally:
                if client_setter:
                    client_setter(None)
                safe_close(client)
            print(messages["client_disconnected"])
        except socket.timeout:
            continue
        except Exception as e:
            if running_check():
                print(messages["server_error"].format(error=e))
    safe_close(server)
    if server_setter:
        server_setter(None)
