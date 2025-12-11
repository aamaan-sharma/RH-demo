import AppRoutes from "./Router";
import { GoogleOAuthProvider } from "@react-oauth/google";

function App() {
  return (
    <GoogleOAuthProvider clientId="1004538351726-me2vaf3cvr8umotevme7ks89mb5rojii.apps.googleusercontent.com">
      <AppRoutes />
    </GoogleOAuthProvider>
  );
}

export default App;
