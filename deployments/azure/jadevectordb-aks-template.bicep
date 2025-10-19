targetScope: resourceGroup

@description('Location for all resources')
param location string = resourceGroup().location

@description('Name for the AKS cluster')
param clusterName string = 'jadevectordb-cluster'

@description('Number of nodes in the AKS node pool')
param agentCount int = 3

@description('VM size for the AKS nodes')
@allowed([
  'Standard_D2_v2'
  'Standard_D4_v2'
  'Standard_D8_v2'
  'Standard_D16_v2'
  'Standard_D32_v2'
  'Standard_D64_v2'
])
param agentVMSize string = 'Standard_D4_v2'

@description('DNS prefix for the AKS cluster')
param dnsPrefix string = 'jadevectordb-${uniqueString(resourceGroup().id)}'

@description('Docker image for JadeVectorDB')
param dockerImage string = 'jadevectordb/jadevectordb:latest'

@description('Kubernetes version for the AKS cluster')
param kubernetesVersion string = '1.27'

// Create storage account
resource storageAccount 'Microsoft.Storage/storageAccounts@2022-09-01' = {
  name: take('jadevectordb${uniqueString(resourceGroup().id)}', 24)
  location: location
  kind: 'StorageV2'
  sku: {
    name: 'Standard_LRS'
  }
  properties: {
    accessTier: 'Hot'
    supportsHttpsTrafficOnly: true
    minimumTlsVersion: 'TLS1_2'
    allowBlobPublicAccess: false
  }
}

// Create AKS cluster
resource aksCluster 'Microsoft.ContainerService/managedClusters@2022-09-01' = {
  name: clusterName
  location: location
  tags: {
    displayName: 'JadeVectorDB AKS Cluster'
  }
  properties: {
    kubernetesVersion: kubernetesVersion
    enableRBAC: true
    dnsPrefix: dnsPrefix
    agentPoolProfiles: [
      {
        name: 'agentpool'
        count: agentCount
        vmSize: agentVMSize
        osType: 'Linux'
        mode: 'System'
        enableAutoScaling: true
        minCount: 1
        maxCount: 10
        type: 'VirtualMachineScaleSets'
      }
    ]
    apiServerAccessProfile: {
      enablePrivateCluster: false
    }
    networkProfile: {
      networkPlugin: 'kubenet'
      loadBalancerSku: 'standard'
    }
  }
  dependsOn: [
    storageAccount
  ]
}

output controlPlaneFQDN string = aksCluster.properties.fqdn
output nodeResourceGroup string = aksCluster.properties.nodeResourceGroup