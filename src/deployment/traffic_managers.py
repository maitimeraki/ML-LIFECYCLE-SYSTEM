import os
# Check if running in Kubernetes
IN_KUBERNETES = os.path.exists("/var/run/secrets/kubernetes.io/serviceaccount/token")

class IstioTrafficManager:
    """Controls traffic splitting via Istio VirtualService."""
    
    def set_traffic_split(self, endpoint: str, split: dict):
        """
        Updates Istio VirtualService weights.
        Called by ModelDeployer._execute_canary().
        """
        if not IN_KUBERNETES:
            print(f"[SKIP] Not in Kubernetes. Would set traffic split: {split}")
            return
        import yaml
        from kubernetes import client, config
        
        # Original K8s code here
        config.load_incluster_config()
        api = client.CustomObjectsApi()
        
        # Build VirtualService patch
        routes = []
        for variant, weight in split.items():
            if variant == "champion":
                routes.append({
                    "destination": {
                        "host": f"{endpoint}-champion",
                        "port": {"number": 8000}
                    },
                    "weight": int(weight)
                })
            elif variant == "challenger":
                routes.append({
                    "destination": {
                        "host": f"{endpoint}-challenger",
                        "port": {"number": 8000}
                    },
                    "weight": int(weight)
                })
        
        patch = {
            "spec": {
                "http": [{"route": routes}]
            }
        }
        
        # Apply patch
        api.patch_namespaced_custom_object(
            group="networking.istio.io",
            version="v1beta1",
            namespace="production",
            plural="virtualservices",
            name=f"{endpoint}-canary",
            body=patch,
        )