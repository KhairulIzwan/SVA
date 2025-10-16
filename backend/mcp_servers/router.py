"""Mock router server for testing"""
class RouterServer:
    def route_request(self, request_type, data):
        return {"status": "routed", "service": "mock", "data": data}