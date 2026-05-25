' Football Predictor launcher.
' If the Streamlit server isn't already running on port 8511, starts it hidden
' in the background, waits for it to come up, then opens the browser.

Option Explicit

Dim shell, http, url, pythonExe, appDir, isRunning, i

Set shell = CreateObject("WScript.Shell")
url = "http://localhost:8511"
pythonExe = shell.ExpandEnvironmentStrings("%LOCALAPPDATA%\Python\bin\python.exe")
appDir = "C:\Users\jadha\football-predictor"

' --- Check if the server is already running ---
isRunning = False
On Error Resume Next
Set http = CreateObject("MSXML2.XMLHTTP")
http.Open "GET", url, False
http.send
If Err.Number = 0 And http.status = 200 Then
    isRunning = True
End If
Err.Clear
On Error Goto 0

' --- If not, start it hidden ---
If Not isRunning Then
    shell.CurrentDirectory = appDir
    shell.Run """" & pythonExe & """ -m streamlit run app.py --server.headless true --server.port 8511 --browser.gatherUsageStats false", 0, False

    ' Wait up to 60s for the server to respond (cold start loads pandas+xgboost)
    For i = 1 To 60
        WScript.Sleep 1000
        On Error Resume Next
        http.Open "GET", url, False
        http.send
        If Err.Number = 0 And http.status = 200 Then
            Err.Clear
            Exit For
        End If
        Err.Clear
        On Error Goto 0
    Next
End If

' --- Open browser ---
shell.Run url, 1, False
