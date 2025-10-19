# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Creates a Google Kubernetes Engine cluster with JadeVectorDB deployment."""

def generate_config(context):
    """Generate deployment configuration."""
    resources = []
    outputs = []

    cluster_name = context.properties.get('clusterName', 'jadevectordb-cluster')
    zone = context.properties.get('zone', 'us-central1-a')
    node_count = context.properties.get('nodeCount', 3)
    node_machine_type = context.properties.get('nodeMachineType', 'n1-standard-4')
    disk_size_gb = context.properties.get('diskSizeGb', 100)
    min_node_count = context.properties.get('minNodeCount', 1)
    max_node_count = context.properties.get('maxNodeCount', 10)
    enable_autoscaling = context.properties.get('enableAutoscaling', True)
    enable_network_policy = context.properties.get('enableNetworkPolicy', False)
    enable_private_nodes = context.properties.get('enablePrivateNodes', False)
    docker_image = context.properties.get('dockerImage', 'jadevectordb/jadevectordb:latest')

    # Create the GKE cluster
    resources.append({
        'name': cluster_name,
        'type': 'container.v1.cluster',
        'properties': {
            'zone': zone,
            'cluster': {
                'name': cluster_name,
                'initialNodeCount': node_count,
                'nodeConfig': {
                    'machineType': node_machine_type,
                    'diskSizeGb': disk_size_gb,
                    'oauthScopes': [
                        'https://www.googleapis.com/auth/cloud-platform'
                    ]
                },
                'addonsConfig': {
                    'horizontalPodAutoscaling': {
                        'disabled': False
                    },
                    'httpLoadBalancing': {
                        'disabled': False
                    },
                    'kubernetesDashboard': {
                        'disabled': True
                    }
                },
                'ipAllocationPolicy': {
                    'useIpAliases': True
                },
                'enableAutoscaling': enable_autoscaling,
                'autoscaling': {
                    'minNodeCount': min_node_count,
                    'maxNodeCount': max_node_count
                },
                'enableNetworkPolicyConfig': enable_network_policy,
                'privateClusterConfig': {
                    'enablePrivateNodes': enable_private_nodes
                } if enable_private_nodes else {}
            }
        }
    })

    # JadeVectorDB Kubernetes deployment
    resources.append({
        'name': 'jadevectordb-k8s-manifest',
        'type': 'gcp-types/k8s-v1:Deployment',
        'properties': {
            'cluster': cluster_name,
            'zone': zone,
            'manifest': {
                'apiVersion': 'apps/v1',
                'kind': 'StatefulSet',
                'metadata': {
                    'name': 'jadevectordb',
                    'namespace': 'default'
                },
                'spec': {
                    'serviceName': 'jadevectordb',
                    'replicas': node_count,
                    'selector': {
                        'matchLabels': {
                            'app': 'jadevectordb'
                        }
                    },
                    'template': {
                        'metadata': {
                            'labels': {
                                'app': 'jadevectordb'
                            }
                        },
                        'spec': {
                            'containers': [
                                {
                                    'name': 'jadevectordb',
                                    'image': docker_image,
                                    'ports': [
                                        {
                                            'containerPort': 8080
                                        },
                                        {
                                            'containerPort': 8081
                                        }
                                    ],
                                    'env': [
                                        {
                                            'name': 'JADE_DB_PORT',
                                            'value': '8080'
                                        },
                                        {
                                            'name': 'JADE_DB_RPC_PORT',
                                            'value': '8081'
                                        },
                                        {
                                            'name': 'JADE_DB_LOG_LEVEL',
                                            'value': 'INFO'
                                        },
                                        {
                                            'name': 'JADE_DB_CLUSTER_SIZE',
                                            'value': str(node_count)
                                        },
                                        {
                                            'name': 'JADE_DB_DATA_DIR',
                                            'value': '/data'
                                        },
                                        {
                                            'name': 'JADE_DB_CONFIG_DIR',
                                            'value': '/config'
                                        }
                                    ],
                                    'volumeMounts': [
                                        {
                                            'name': 'data-volume',
                                            'mountPath': '/data'
                                        },
                                        {
                                            'name': 'config-volume',
                                            'mountPath': '/config'
                                        }
                                    ],
                                    'livenessProbe': {
                                        'httpGet': {
                                            'path': '/health',
                                            'port': 8080
                                        },
                                        'initialDelaySeconds': 60,
                                        'periodSeconds': 30
                                    },
                                    'readinessProbe': {
                                        'httpGet': {
                                            'path': '/health',
                                            'port': 8080
                                        },
                                        'initialDelaySeconds': 30,
                                        'periodSeconds': 10
                                    }
                                }
                            ],
                            'volumes': [
                                {
                                    'name': 'config-volume',
                                    'emptyDir': {}
                                }
                            ]
                        }
                    },
                    'volumeClaimTemplates': [
                        {
                            'metadata': {
                                'name': 'data-volume'
                            },
                            'spec': {
                                'accessModes': ['ReadWriteOnce'],
                                'resources': {
                                    'requests': {
                                        'storage': '100Gi'
                                    }
                                }
                            }
                        }
                    ]
                }
            }
        },
        'metadata': {
            'dependsOn': [cluster_name]
        }
    })

    # Service to expose JadeVectorDB
    resources.append({
        'name': 'jadevectordb-service',
        'type': 'gcp-types/k8s-v1:Service',
        'properties': {
            'cluster': cluster_name,
            'zone': zone,
            'manifest': {
                'apiVersion': 'v1',
                'kind': 'Service',
                'metadata': {
                    'name': 'jadevectordb-service',
                    'namespace': 'default'
                },
                'spec': {
                    'selector': {
                        'app': 'jadevectordb'
                    },
                    'ports': [
                        {
                            'protocol': 'TCP',
                            'port': 8080,
                            'targetPort': 8080
                        },
                        {
                            'protocol': 'TCP',
                            'port': 8081,
                            'targetPort': 8081
                        }
                    ],
                    'type': 'LoadBalancer'
                }
            }
        },
        'metadata': {
            'dependsOn': [cluster_name]
        }
    })

    # Add outputs
    outputs.extend([
        {
            'name': 'clusterName',
            'value': '$(ref.' + cluster_name + '.name)'
        },
        {
            'name': 'clusterEndpoint',
            'value': '$(ref.' + cluster_name + '.endpoint)'
        },
        {
            'name': 'clusterCaCertificate',
            'value': '$(ref.' + cluster_name + '.masterAuth.clusterCaCertificate)'
        },
        {
            'name': 'serviceName',
            'value': '$(ref.jadevectordb-service.name)'
        },
        {
            'name': 'serviceIP',
            'value': '$(ref.jadevectordb-service.status.loadBalancer.ingress[0].ip)'
        }
    ])

    return {
        'resources': resources,
        'outputs': outputs
    }