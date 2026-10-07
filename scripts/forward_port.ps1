# Forwards this Windows PC's port to the Awale server in WSL2, so other machines can reach it,
# over IPv4 (local network) and IPv6 (internet: the box's IPv4 is CGNAT, so it cannot forward ports).
# Run as administrator, again after each reboot (the WSL address changes):
#
#   powershell -ExecutionPolicy Bypass -File forward_port.ps1 [-Port 8000] [-Distro Ubuntu-20.04] [-Remove]
#
# The server must listen on all WSL interfaces: uv run python -m core.server --host 0.0.0.0
param(
    [int]$Port = 8000,
    [string]$Distro = "",  # The WSL distribution running the server, default one if empty
    [switch]$Remove
)

$rule = "Awale server $Port"
netsh interface portproxy delete v4tov4 listenport=$Port listenaddress=0.0.0.0 | Out-Null
netsh interface portproxy delete v6tov4 listenport=$Port listenaddress=:: | Out-Null
Remove-NetFirewallRule -DisplayName $rule -ErrorAction SilentlyContinue
if ($Remove) {
    Write-Host "Removed the forward of port $Port"
    exit
}

$wslArgs = if ($Distro) { @("-d", $Distro) } else { @() }
$wslIp = (wsl @wslArgs hostname -I).Trim().Split(" ")[0]
netsh interface portproxy add v4tov4 listenport=$Port listenaddress=0.0.0.0 connectport=$Port connectaddress=$wslIp
netsh interface portproxy add v6tov4 listenport=$Port listenaddress=:: connectport=$Port connectaddress=$wslIp
New-NetFirewallRule -DisplayName $rule -Direction Inbound -Protocol TCP -LocalPort $Port -Action Allow -Profile Any | Out-Null  # Any: Windows may class the home network as Public
Write-Host "Port $Port of this PC (IPv4 and IPv6) now forwards to WSL at ${wslIp}:$Port, firewall open"
netsh interface portproxy show all
$ipv6 = Get-NetIPAddress -AddressFamily IPv6 -PrefixOrigin RouterAdvertisement -AddressState Preferred |
    Where-Object { $_.PreferredLifetime -gt [TimeSpan]::FromDays(2) } |  # Temporary addresses last a day
    Select-Object -First 1 -ExpandProperty IPAddress
if ($ipv6) { Write-Host "Internet URL: http://[${ipv6}]:$Port/<token>/  (open this port to $ipv6 in the box's IPv6 firewall)" }
