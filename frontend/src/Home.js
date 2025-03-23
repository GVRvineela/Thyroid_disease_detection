// import React from 'react'

// function Home() {
//     return (
//         <div>Home</div>
//     )
// }

// export default Home


import React from 'react';

function Home() {
    const redirectToHomePage = () => {
        window.location.href = 'http://127.0.0.1:5500/after.html';
    };

    return (
        <div>
            {/* 
                You may call redirectToHomePage() when the component mounts,
                or you can trigger it based on user interaction (e.g., onClick).
            */}
            {redirectToHomePage()}
        </div>
    );
}

export default Home;


// import React from 'react';

// function Home() {
//     const redirectToHomePage = () => {
//         // Update the URL to redirect to index.html upon signing in
//         window.location.href = 'http://localhost:3000/index.html';
//     };

//     // Call redirectToHomePage when the component mounts
//     React.useEffect(() => {
//         redirectToHomePage();
//     }, []);

//     // No need to return any JSX here since we are redirecting
//     return null;
// }

// export default Home;