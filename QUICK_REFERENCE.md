# FaceFetch Authentication - Quick Reference Card

## 🚀 5-Second Quick Start

```bash
python web_server.py
# Visit: http://localhost:5000
# Create account → Login → Done! 🎉
```

## 📍 Where to Click

```
http://localhost:5000
    ↓
"Sign up here" → Register → Confirm Email?
                              ↓
                           Login Page
                              ↓
Login → Password? → Successful! → Dashboard
```

## 🔐 User/Password Requirements

| Field | Requirement |
|-------|-------------|
| First Name | Any text |
| Last Name | Any text |
| Email | Valid format (user@domain.com) |
| Password | At least 8 characters |
| Confirm Password | Must match password |
| Company | Optional |
| Terms | Must agree |

## 🎯 What Each Button Does

| Button | Location | Action |
|--------|----------|--------|
| "Sign In" | Login Page | Login with email/password |
| "Sign up here" | Login Page | Go to registration |
| "Create Account" | Register Page | Create new account |
| "Logout" | Dashboard Header | Logout & clear session |
| "Forgot password" | Login Page | (UI ready, not implemented) |

## 🔑 Test Credentials

After creating an account, you can login with:
- **Email**: The email you registered with
- **Password**: The password you chose

Example:
```
Email: john@example.com
Password: MyPassword123!
```

## 📊 File Locations

| What | Where |
|-----|-------|
| Login Form | `templates/login_page.html` |
| Register Form | `templates/register_page.html` |
| Dashboard | `templates/index.html` |
| Database | `facefetch_users.db` |
| Web Server Code | `web_server.py` |
| Tests | `test_auth.py` |

## 🔗 API Endpoints Quick Reference

### Public (No Login Required)
```
POST   /login              → Login
POST   /register           → Register
GET    /logout             → Logout
GET    /api/auth/status    → Check if logged in
```

### Protected (Login Required)
```
GET    /                   → Dashboard
GET    /api/user/profile   → Your profile
GET    /video_feed         → Live video
GET    /api/status         → System status
GET    /api/history        → Detections log
GET    /api/alerts         → Security alerts
```

## 🧪 Quick Test Commands

### Test with cURL

**Register:**
```bash
curl -X POST http://localhost:5000/register \
  -d "firstName=Test&lastName=User&email=test@example.com&password=TestPass123"
```

**Login:**
```bash
curl -c cookies.txt -X POST http://localhost:5000/login \
  -d "email=test@example.com&password=TestPass123"
```

**Get Profile (requires login):**
```bash
curl -b cookies.txt http://localhost:5000/api/user/profile
```

### Test with Browser
1. Open `http://localhost:5000`
2. You'll be on login page
3. Register or login
4. Try clicking different sections

### Automated Tests
```bash
python test_auth.py
```

## 🐛 Common Issues & Quick Fixes

| Problem | Solution |
|---------|----------|
| "Port 5000 in use" | Change port in web_server.py or kill process |
| "Email already registered" | Use different email or delete from database |
| "Invalid password" | Password must be 8+ characters |
| "Session expired" | Login again |
| "Passwords don't match" | Make sure both fields are identical |

## 🔒 Password Security Tips

✅ DO:
- Use 8+ characters
- Mix upper and lowercase
- Include numbers
- Include special characters
- Use unique passwords

❌ DON'T:
- Use "password123"
- Use birthdates
- Reuse passwords
- Share passwords
- Write down passwords

## 📖 Documentation Map

**Choose your path:**

```
Just want to use it?
    ↓
→ QUICK_START.md

Building an integration?
    ↓
→ AUTHENTICATION_GUIDE.md

Want to understand the code?
    ↓
→ AUTH_IMPLEMENTATION_SUMMARY.md

Interested in system design?
    ↓
→ ARCHITECTURE_DIAGRAMS.md

Checking features?
    ↓
→ IMPLEMENTATION_CHECKLIST.md

Complete overview?
    ↓
→ AUTHENTICATION_README.md
```

## 🎮 Interactive Demo

### Step 1: Start Server
```bash
python web_server.py
```

### Step 2: Register
1. Open `http://localhost:5000`
2. Click "Sign up here"
3. Fill in:
   - First Name: John
   - Last Name: Doe
   - Email: john@example.com
   - Password: MyPassword123!
   - Confirm: MyPassword123!
4. Click "Create Account"

### Step 3: Login
1. Email: john@example.com
2. Password: MyPassword123!
3. Click "Sign In"

### Step 4: Explore
- See your name in header
- Click logout button
- Try logging in again

## 💾 Database Quick Commands

### View all users
```bash
sqlite3 facefetch_users.db "SELECT email, first_name, last_name FROM users;"
```

### Delete a user
```bash
sqlite3 facefetch_users.db "DELETE FROM users WHERE email='test@example.com';"
```

### Start fresh
```bash
# Remove database (recreates automatically)
rm facefetch_users.db
# Restart app
python web_server.py
```

## 🔑 Environment Variables (Optional)

For production, set these:
```bash
export SECRET_KEY="your-secret-key"
export FLASK_ENV="production"
```

## 📱 Responsive Design

Works on:
- ✅ Desktop (1920px+)
- ✅ Tablet (768px-1024px)
- ✅ Mobile (320px-767px)

Try resizing browser to test!

## ⚡ Performance Notes

- **Database**: SQLite (fine for small apps)
- **Password Hashing**: Intentionally slow (security feature)
- **Video Stream**: H.264 MJPEG codec
- **Typical Load**: <100ms per request

## 🔄 User Session Lifecycle

```
Login
  ↓
Session Created (secure cookie)
  ↓
User Browsing (session checked on each request)
  ↓
Click Logout
  ↓
Session Cleared
  ↓
Redirect to Login
```

## ✅ Status Check

Is everything working?

```
Can start app?          ✓
Can visit login?        ✓
Can register?           ✓
Can login?              ✓
Can see dashboard?      ✓
Can logout?             ✓
Can access APIs?        ✓
Can run tests?          ✓
```

If all ✓, you're good to go! 🚀

## 🎓 Security in 30 Seconds

1. **Passwords**: Bcrypt hashed (cannot be reversed)
2. **Sessions**: Encrypted cookies (can't be tampered)
3. **APIs**: Check session on every request
4. **Validation**: Check input on server and client
5. **HTTPS**: Ready for secure connection

## 🆘 Need Help?

| Question | Answer |
|----------|--------|
| How do I...? | QUICK_START.md |
| What's the API? | AUTHENTICATION_GUIDE.md |
| Show me code | AUTH_IMPLEMENTATION_SUMMARY.md |
| How does it work? | ARCHITECTURE_DIAGRAMS.md |
| Is it done? | IMPLEMENTATION_CHECKLIST.md |

## 🚀 Deploy to Production

Before going live:

- [ ] Set SECRET_KEY environment variable
- [ ] Enable HTTPS (set Secure flag)
- [ ] Backup database
- [ ] Set up monitoring
- [ ] Test with real load
- [ ] Review error logs

## 💡 Pro Tips

💡 **Use "Remember Me"** → Saves email automatically
💡 **Strong Password** → Use upper, lower, number, special char
💡 **Browser DevTools** → Check Console for errors
💡 **Test Script** → Run `test_auth.py` to validate
💡 **Read Docs** → Each doc has specific info

## 🎯 Success Checklist

- [ ] App running (`python web_server.py`)
- [ ] Can visit login page
- [ ] Can create account
- [ ] Can login successfully
- [ ] Can see dashboard
- [ ] Can logout
- [ ] Tests passing
- [ ] Ready to use!

---

## 📞 Quick Links

- **Documentation**: See folder for 6 guide files
- **Tests**: `python test_auth.py`
- **Server**: `python web_server.py`
- **Browser**: `http://localhost:5000`

**Happy coding! 🚀**

